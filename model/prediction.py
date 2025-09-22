import numpy as np

import torch

from torch.utils.data import DataLoader
from model.dataset import TimeSeriesDataset

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# ---------- prediction helpers for BCE heads ----------

def predict_proba(model, X, mask, device=None):
	""" return per-patient probabilities for the 3 tasks """
	with torch.no_grad():
		device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		ds = TimeSeriesDataset(X, np.zeros((X.shape[0], 3)), mask)
		loader = DataLoader(ds, batch_size=128, shuffle=False)
		model.to(torch.device(device)).eval()

		probs_all = []
		for xb, _, _, lb in loader: # Loops over batches
			xb, lb = xb.to(device), lb.to(device)
			z, logits, _ = model(xb, lb)
			probs = torch.sigmoid(torch.stack(logits, dim=1))  # (B,3)
			probs_all.append(probs.cpu())
		return torch.cat(probs_all, dim=0).numpy()

def predict_logits(model, X, mask, device=None):
	""" return per-patient logits for the 3 tasks """
	with torch.no_grad():
		device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		ds = TimeSeriesDataset(X, np.zeros((X.shape[0], 3)), mask)
		loader = DataLoader(ds, batch_size=128, shuffle=False)
		model.to(torch.device(device)).eval()

		logits_all = []
		for xb, _, _, lb in loader: # Loops over batches
			xb, lb = xb.to(device), lb.to(device)
			z, logits, _ = model(xb, lb)
			L = torch.stack(logits, dim=1)  # (B,3)
			logits_all.append(L.cpu())
		return torch.cat(logits_all, dim=0).numpy()

# ---------- prediction helpers for KNN heads ----------

def encode_embeddings(model, X, mask, device=None, batch_size=128):
	""" Returns embeddings for each patient """
	with torch.no_grad():
		device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		model.to(torch.device(device)).eval()

		# dummy y just to satisfy the Dataset signature
		ds = TimeSeriesDataset(X, np.zeros((X.shape[0], 3), dtype=np.float32), mask)
		loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
		
		embs = [[],[],[]]
		for xb, _, _, lb in loader:
			xb, lb = xb.to(device), lb.to(device)
			z, _, _ = model(xb, lb) # (B, Z)
			for i in range(3):
				e = model.project(z,i) # (B, Zp) -> the SupCon space
				embs[i].append(e.cpu())
		
		full_embs = [torch.cat(embs[i], dim=0).numpy() for i in range(3)]
		return full_embs


def predict_proba_knn(model, X_train, X_test, mask_train, mask_test, y_train, n_neig=10, device=None):
	""" return per-patient probabilities for the 3 tasks using KNN in the embeddings """
	probs = []
	# find embeddings (fit index)
	Ztr = encode_embeddings(model, X_train, mask_train)
	Zte = encode_embeddings(model, X_test,  mask_test)

	y_tr = y_train.astype(np.float64)

	for k in range(3):
		# cosine on unit sphere; Euclidean works too when normalized
		knn = NearestNeighbors(n_neighbors=n_neig, metric="cosine")
		knn.fit(Ztr[k])

		# For each test point find its nearest train points
		dist, idx = knn.kneighbors(Zte[k], n_neighbors=n_neig)
		dist = dist.astype(np.float64)

		# inverse-distance-weighted voting (smaller dist => larger weight)
		w = 1.0 / np.maximum(dist, 1e-12)
		w_sum = w.sum(axis=1, keepdims=True)
		w = np.divide(w, w_sum, out=np.zeros_like(w), where=w_sum > 0)

		pt = (y_tr[idx, k] * w).sum(axis=1)		   # weighted mean in float64
		pt = np.clip(pt, 0.0, 1.0)					 # kill 1.0000001 etc.
		probs.append(pt.astype(np.float32))

	return np.vstack(probs).T # shape (N_test, 3)


# ---------- calibration ----------

def fit_calibrators(val_logits, val_labels):
	""" Platt scaling calibrator - 
		learns a logistic regression model for the given prediction head - maps model’s predictions -> calibrated probability """
	calibrators = []
	for k in range(val_logits.shape[1]):  # 3 tasks
		lr = LogisticRegression(solver="lbfgs")
		# sklearn expects 2D features; feed the single logit as a column
		lr.fit(val_logits[:, [k]], val_labels[:, k])
		calibrators.append(lr)
	return calibrators

def predict_proba_calibrated(calibrators, test_logits):
	""" Apply fitted Platt scalers to map logits to calibrated probabilities """
	cols = []
	for k, lr in enumerate(calibrators):
		p1 = lr.predict_proba(test_logits[:, [k]])[:, 1]
		cols.append(p1)
	return np.column_stack(cols)

def fit_isotonic_calibrators(val_probs, val_labels):
	""" Platt scaling calibrator - 
		learns a logistic regression model for the given prediction head - maps model’s predictions -> calibrated probability """
	cals = []
	for k in range(val_probs.shape[1]):
		ir = IsotonicRegression(out_of_bounds="clip")
		ir.fit(val_probs[:, k], val_labels[:, k])
		cals.append(ir)
	return cals

def predict_isotonic_proba_calibrated(calibrators, test_probs):
	cols = []
	for k, ir in enumerate(calibrators):
		cols.append(ir.transform(test_probs[:, k]))
	return np.column_stack(cols)