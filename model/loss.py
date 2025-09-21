import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Losses ----------

def masked_mse(recon, x, mask):
	""" MSE average reconstruction error (ignoring padded so loss scale doesn’t depend on sequence length)
	recon, x: (B,T,D); mask: (B,T)
	 * B = batch size
	 * T = number of time steps per patient (after padding)
	 * D = number of features """
	mse = F.mse_loss(recon, x, reduction="none") # squared error per feature (B,T,D)
	mse = mse.sum(dim=-1) # sum error over features (B,T)
	denom = mask.sum() * x.shape[-1] + 1e-8 # 1e-8 to avoid division by zero
	return (mse * mask).sum() / denom # total error avg by the number of features per timestep
	

class SupConLoss(nn.Module):
	""" Supervised Contrastive Loss, for one task, over L2-normalised (similarity depends only on direction) embeddings z (follows from Khosla et al) - 
	Goal is so that similar patients (same label) be close in latent space, and patients with different labels be far apart.
	
	anchor_mode:
	  - "both"       : all samples are anchors (standard SupCon)
	  - "positives"  : only y==1 rows are anchors (useful for rare positives) 
	temperature: controls how “peaky” the softmax is """
	def __init__(self, temperature=0.2, anchor_mode="both"):
		super().__init__()
		self.temp = temperature
		self.anchor_mode = anchor_mode
	
	def forward(self, z, y):
		""" z : (B, D) embeddings that will be L2-normalized 
		    y : (B,)   labels in {0,1} for this ONE task """
		
		B = z.size(0)
		# cosine similarity matrix
		sim = z @ z.t() / self.temp
		# mask out self-similarity when computing contrastive loss - we compare each sample with others in the batch
		logits_mask = 1.0 - torch.eye(B, device=z.device) # flip identity matrix -> 1 for j!=i, 0 for self

		# positive-pair mask - same label, excluding self
		y = y.view(-1,1)
		pos_mask = (y == y.t()).float() * logits_mask
		
		# decide which samples in the batch act as anchors
		if self.anchor_mode == "positives": # focuses the contrastive loss on the rare important cases
			anchors_mask = (y.view(-1) == 1) # true for pos only
		else:  # "both" - classic SupCon setup
			anchors_mask = torch.ones(B, dtype=torch.bool, device=z.device)

		# numerical stability: subtract per-row max before exp
		sim = sim - sim.max(dim=1, keepdim=True).values
		exp_sim = torch.exp(sim) * logits_mask
		# calc how likely is j to be a neighbor of i compared to all others
		log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))
		
		# for each anchor average over its positives
		pos_count = pos_mask.sum(dim=1)
		loss_i = -(pos_mask * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
		
		# valid anchors are both selected and have >=1 positive partner
		valid = anchors_mask & (pos_count > 0)
		if valid.any():
			return loss_i[valid].mean()
		else:
			return z.new_tensor(0.0) # no valid anchors in batch