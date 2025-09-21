import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ---------- Dataset ----------
class TimeSeriesDataset(Dataset):
	""" X: (N, T, D), y: (N, 3), mask: (N, T)
	where - 
	* N = number of patients (subjects)
	* T = maximum number of time steps per patient
		(e.g., we use 48h with 6-hour bins → T = 8 time steps).
	* D = number of features per time step """
	
	def __init__(self, X, y, mask):
		self.X = torch.tensor(X, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32)
		self.mask = torch.tensor(mask, dtype=torch.float32)
		self.lengths = self.mask.sum(dim=1).long()

	def __len__(self): return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx], self.mask[idx], self.lengths[idx]

# ---------- Balanced batch sampler ----------
class BalancedPositivesPerTaskSampler(Sampler):
	""" Ensures each batch has at least `p_per_task` positive examples for each of the 3 binary tasks. 
	Oversamples (with replacement) if the pool is small and remaining slots filled randomly. """
	def __init__(self, y, batch_size=64, p_per_task=4, seed=42):
		self.y = np.asarray(y)  # (N,3)
		self.N = len(self.y)
		self.batch_size = batch_size
		self.p_per_task = p_per_task # how many positives
		self.rng = np.random.RandomState(seed)

		# indices of positives per task
		self.pos_idx = [np.where(self.y[:, k] == 1)[0] for k in range(self.y.shape[1])] # subjects that are positive for each task
		self.all_idx = np.arange(self.N)

	def __iter__(self):
		# Simple infinite pass until we've covered the dataset roughly once
		num_batches = math.ceil(self.N / self.batch_size)
		for _ in range(num_batches): # for each batch 
			chosen = []
			for t in range(3): # for each task
				pool = self.pos_idx[t]
				if len(pool) == 0:
					continue  # no positives; skip
				if len(pool) >= self.p_per_task: 
					# if we have enough positives to sample from -> sample with no replacement
					pick = self.rng.choice(pool, size=self.p_per_task, replace=False)
				else:
					# if not enough positives exist -> oversample with replacement
					pick = self.rng.choice(pool, size=self.p_per_task, replace=True)  # oversample
				chosen.extend(pick.tolist())

			# fill the rest of the batch with random patients (negatives or extra positives) with no replacement
			need = max(0, self.batch_size - len(chosen))
			fill = self.rng.choice(self.all_idx, size=need, replace=False).tolist()
			batch = chosen + fill
			
			yield batch

	def __len__(self):
		# Number of batches per epoch (roughly one dataset pass)
		return math.ceil(self.N / self.batch_size)
