import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from model.dataset import TimeSeriesDataset, BalancedPositivesPerTaskSampler
from model.model import MultiTaskSeqGRUAE
from model.loss import masked_mse, SupConLoss, focal_bce_with_logits

import os, random

def set_seed(seed: int = 42):
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	# deterministic (slower but repeatable)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# optional, stricter determinism:
	# torch.use_deterministic_algorithms(True)

# ---------- Training ----------
def train_multitask_seq_ae(
		X, y, mask,
		input_dim,
		batch_size=64,
		p_per_task=4,
		epochs=20,
		scale_warmup_epochs=0,
		lr_warmup_epochs=8,

		warmup_scales={'lambda_recon':0.0, 'lambda_bce':0.0, 'lambda_supcon':0.0},
		scales={'lambda_recon':1.0, 'lambda_bce':1.0, 'lambda_supcon':0.5},

		warmup_lrs={'AE':0.5,'BCE':[1.0,1.0,1.2],'SupCon':0.5},
		lrs={'AE':5e-4, 'BCE':[5e-4,5e-4,6e-4], 'SupCon':2e-3},# {'AE':j[0], 'BCE':j[1:4], 'SupCon':j[4]},#

		latent_dim=80, 
		SupCon_latent_dim=16,

		pooling_mode = "mean+final", # "final", "mean+final", "mean+max+final", "mean+attn"

		weights_bce=[1,1,1],
		weights_supcon=[1,1,1],
		temperature=0.07,
		supcon_gamma=0.7,   # CB exponent (0.5–0.8 is gentle)
		supcon_delta=0.5,   # inverse-anchors exponent (0.3–0.7)

		device="cuda" if torch.cuda.is_available() else "cpu",
		seed=0):
	""" this function trains the multi-task GRU autoencoder with three losses at once:
	 * Reconstruction (masked MSE)
	 * Classification for 3 tasks (BCE)
	 * Contrastive (SupCon) in a projected space 
	returns the trained model """
	
	# data setup
	ds = TimeSeriesDataset(X, y, mask)
	set_seed(seed)
	sampler = BalancedPositivesPerTaskSampler(y, batch_size=batch_size, p_per_task=p_per_task, seed=seed)
	loader = DataLoader(ds, batch_sampler=sampler)
	
	weights_bce = torch.as_tensor(weights_bce, dtype=torch.float32, device=device)
	weights_supcon = torch.as_tensor(weights_supcon, dtype=torch.float32, device=device)
	
	# def model and loss objects
	model = MultiTaskSeqGRUAE(input_dim=input_dim, latent_dim=latent_dim, SupCon_latent_dim=SupCon_latent_dim, pooling=pooling_mode).to(device)
	opt = torch.optim.Adam([
		{"params": model.encoder.parameters(), "lr": lrs['AE']},
		{"params": model.decoder.parameters(), "lr": lrs['AE']},
		{"params": model.cls_heads[0].parameters(), "lr": lrs['BCE'][0]}, # mortality
		{"params": model.cls_heads[1].parameters(), "lr": lrs['BCE'][1]}, # prolonged
		{"params": model.cls_heads[2].parameters(), "lr": lrs['BCE'][2]}, # READMISSION
		{"params": model.proj_heads.parameters(), "lr": lrs['SupCon']},
	])

	# numerically stable BCE (Binary Cross-Entropy) with per-task class weights to balance rare posives
	bce_losses = [nn.BCEWithLogitsLoss(pos_weight=weights_bce[k]) for k in range(3)]
	
	running_total = {"recon": [], "bce": [], "supcon": [], "total": []}
	for ep in range(1, epochs + 1): # epoch loop
		model.train()
		running = {"recon": 0.0, "bce": 0.0, "supcon": 0.0, "total": 0.0}
		n_batches = 0

		# setup weights and scales for wormup
		scale_warmup_phase = (ep <= scale_warmup_epochs)
		cur_lambda_recon, cur_lambda_bce, cur_lambda_supcon = ((warmup_scales['lambda_recon'], warmup_scales['lambda_bce'], warmup_scales['lambda_supcon']) if scale_warmup_phase else (scales['lambda_recon'], scales['lambda_bce'], scales['lambda_supcon']))

		lr_warmup_phase = warmup_lrs if ep <= lr_warmup_epochs else lrs
		lrs_ = [lr_warmup_phase['AE'], lr_warmup_phase['AE'], lr_warmup_phase['BCE'][0], lr_warmup_phase['BCE'][1], lr_warmup_phase['BCE'][2], lr_warmup_phase['SupCon']]
		for pg, lr in zip(opt.param_groups, lrs_):
			pg['lr'] = lr

		# SupCon per task, positives-only anchors for the rare targets mort and read
		sup_5050 = SupConLoss(temperature=temperature, anchor_mode="both")	   # ~50/50
		sup_rare = SupConLoss(temperature=temperature, anchor_mode="positives")  # rare
	
		for xb, yb, mb, lb in loader: # batch loop 
			# xb = inputs (B,T,D), yb = labels (B,3), mb = mask (B,T), lb = lengths (B,)
			xb, yb, mb, lb = xb.to(device), yb.to(device), mb.to(device), lb.to(device)
			
			# forward pass
			opt.zero_grad()
			z, logits, x_hat = model(xb, lb) # returns base latent, target prediction, reconstruction

			# recon loss
			loss_recon = masked_mse(x_hat, xb, mb) * cur_lambda_recon
			
			# BCEWithLogits loss per task and average them
			loss_bce = 0.0
			for k in range(3):
				loss_bce += bce_losses[k](logits[k].view(-1), yb[:,k])
			loss_bce = loss_bce * (cur_lambda_bce / 3.0)
			
			# automaticaly append prolonged_stay (~50/50)
			sup_terms = [sup_5050(model.project(z, 1), yb[:, 1])]
			task_ids = [1]
			# for each of the rare cases append only if there are anchors (pos samples) in the batch 
			for i in [0,2]:
				if (yb[:, i].sum() >= 2):
					sup_terms.append(sup_rare(model.project(z, i), yb[:, i]))
					task_ids.append(i)

			# gentle class-imbalance weights from pos_weights
			cb = weights_supcon[task_ids].clamp_min(1e-12).pow(supcon_gamma)

			# Dynamic per-batch inverse anchors — soften via delta
			invP = torch.tensor(
				[1.0 / max(int((yb[:, t] == 1).sum().item()), 1) for t in task_ids],
				device=z.device, dtype=torch.float32).clamp_min(1e-12).pow(supcon_delta)

			# Combine and normalize
			W = (cb * invP)
			W = W / (W.sum() + 1e-12)
			loss_sup = cur_lambda_supcon * torch.stack(sup_terms).mul(W).sum()

			# combine total loss and backprop
			loss = loss_recon + loss_bce + loss_sup
			loss.backward()
			
			# Gradient clipping keeps training stable (prevents exploding grads)
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
			opt.step()

			running["recon"] += loss_recon.item()
			running["bce"] += loss_bce.item()
			running["supcon"] += loss_sup.item()
			running["total"] += loss.item()
			n_batches += 1

		for k in running: running[k] /= max(1, n_batches)
		running_total["recon"].append(running["recon"])
		running_total["bce"].append(running["bce"])
		running_total["supcon"].append(running["supcon"])
		running_total["total"].append(running["total"])
		# print(f"[Epoch {ep:02d}] total={running['total']:.4f}  recon={running['recon']:.4f}  "f"bce={running['bce']:.4f}  supcon={running['supcon']:.4f}")

	return model, running_total