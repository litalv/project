import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---------- Model ----------
class MultiTaskSeqGRUAE(nn.Module):
	def __init__(self, input_dim, enc_hidden=128, enc_layers=1, dec_hidden=128, dec_layers=1, latent_dim=64, SupCon_latent_dim=32, pooling="final", dropout=0.1):
		""" input_dim : number of features per timestep (D)
		   enc_hidden: hidden size of the encoder GRU
		   dec_hidden: hidden size of the decoder GRU
		   enc_layers: num of stacked GRU layers in the encoder
		   dec_layers: num of stacked GRU layers in the decoder
		   latent_dim: size of the shared patient latent vector z 
		"""
		super().__init__()
		
		# e.g., "final", "mean+final", "mean+max+final", "mean+attn"
		self.pooling = pooling  
		self.modes = [m.strip().lower() for m in self.pooling.split('+')]
		
		# encoder: GRU reads X's timeline step-by-step and compresses it into latent space
		self.encoder = nn.GRU(input_dim, enc_hidden, enc_layers, batch_first=True, dropout=dropout if enc_layers > 1 else 0.0, bidirectional=False)

		# turn the encoder’s hidden vector into latent size
		self.to_latent = nn.Linear(len(self.modes) * enc_hidden, latent_dim)

		# projection head for contrastive learning - 2-layer MLP for each task 
		self.proj_heads = nn.ModuleList([
			nn.Sequential(
				nn.Linear(latent_dim, SupCon_latent_dim), 
				nn.ReLU(inplace=True), 
				nn.Linear(SupCon_latent_dim, SupCon_latent_dim)) 
			for _ in range(3)])

		# three classifiers (separate linear heads - one per task) from shared latent z
		self.cls_heads = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in range(3)])

		# turn the patient representation in z (B, latent_dim) into the decoder’s start state
		self.z_to_h0 = nn.Linear(latent_dim, dec_layers * dec_hidden)
		
		# decoder: reconstruct X from z with GRU -> from vector per patient, rebuild the whole sequence X̂ (B, T, D)
		self.decoder = nn.GRU(input_size=latent_dim, hidden_size=dec_hidden, num_layers=dec_layers, batch_first=True, dropout=dropout if dec_layers > 1 else 0.0)
		
		# reconstruct original features from GRU hidden state 
		self.out = nn.Linear(dec_hidden, input_dim)

	def encode(self, x, lengths):
		# x: (B,T,D), lengths: (B,)
		packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
		out_packed, hN = self.encoder(packed)              # hN: (layers,B,H)
		out, _ = pad_packed_sequence(out_packed, batch_first=True)  # out: GRU outputs for all time steps, (B,T,H)
		B, T, H = out.shape
		device = out.device

		# build (B,T,1) valid mask from lengths
		t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T) # (B,T)
		t_valid = (t_idx < lengths.unsqueeze(1)).float().unsqueeze(-1) # mask for valid time steps, (B,T,1)

		vecs = []
		for mode in self.modes:
			# GRU’s memory after the last real time step - "summary"
			if mode == "final":
				vecs.append(hN[-1]) # (B,H) final hidden state
			# average of the GRU’s outputs across all valid time steps per-patient
			elif mode == "mean":
				summed = (out * t_valid).sum(dim=1) # (B,H), zeroes out padded steps so they don’t affect the average and sum
				denom  = t_valid.sum(dim=1).clamp_min(1.0) # (B,1), how many real steps each patient has
				vecs.append(summed / denom) # (B,H)
			# maximum activation across time for each feature per-patient
			elif mode == "max":
				# t_valid should be (B, T, 1). Make it boolean and DON'T squeeze.
				mask_pad = (t_valid == 0) if t_valid.dtype != torch.bool else ~t_valid
				# Broadcasts (B,T,1) -> (B,T,H)
				masked = out.masked_fill(mask_pad, float("-inf"))
				vecs.append(masked.max(dim=1).values)   # (B, H)
			else:
				raise ValueError(f"Unknown pooling mode: {mode}")

		z_pre = torch.cat(vecs, dim=1) # (B, concat_mult*H)
		z = self.to_latent(z_pre) # (B, latent_dim)
		return z

	def decode(self, z, T):
		""" z: patient embedding for the batch (B, latent_dim)
		    T: number of timesteps to reconstruct """
		# repeat z across time
		B, Z = z.shape
		z_seq = z.unsqueeze(1).repeat(1, T, 1)  # (B,T,Z) with copies of z at every timestep (keeps a constant patient fingerprint at each step)
		
		# map z to the GRU’s initial hidden state
		h0 = self.z_to_h0(z).view(self.decoder.num_layers, B, self.decoder.hidden_size).contiguous()
		# run decoder GRU over time
		dec_out, _ = self.decoder(z_seq, h0) # (B,T,Hd)
		# linear layer converts each timestep’s hidden state to D features
		x_hat = self.out(dec_out) # (B,T,D)
		return x_hat

	def forward(self, x, lengths):
		B, T, _ = x.shape
		z = self.encode(x, lengths)
		# logits per task - 3 × (B,)
		logits = [head(z).squeeze(-1) for head in self.cls_heads] 
		x_hat = self.decode(z, T)
		return z, logits, x_hat
		
	def project(self, z, k):
		e = self.proj_heads[k](z)
		# L2 normalize keeps only “shape/direction” -> so patients with similar outcomes cluster together even if their raw feature scales differ
		return F.normalize(e, p=2, dim=1)
