import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, roc_curve, precision_recall_curve, confusion_matrix, precision_score, ConfusionMatrixDisplay

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def eval_multitask_from_probs(y_true, probs, plot=True, tr=0.5, task_names = ["prolonged_stay", "mortality", "readmission"]):
	"""
	y_true, probs: shape (N, 3) — per-patient labels and predicted probabilities.
	Returns dict of metrics per task. If plot=True, draws ROC, PR, and confusion matrix.
	"""
	report = {}

	for t, name in enumerate(task_names):
		yt = y_true[:, t]
		pt_probs = probs[:, t]

		# threshold-free metrics
		roc_auc = roc_auc_score(yt, pt_probs)
		fpr, tpr, thr = roc_curve(yt, pt_probs) # FPR and TPR across thresholds for ROC ploting
		pr_auc  = average_precision_score(yt, pt_probs)
		prec, rec, _  = precision_recall_curve(yt, pt_probs) # precision and recall across thresholds

		# threshold metrics -> needs binary predictions
		pt = (pt_probs >= tr).astype(int)

		acc = accuracy_score(yt, pt) # share of correct predictions
		f1 = f1_score(yt, pt) # balancing false positives/negatives
		ppv = precision_score(yt, pt)

		report[name] = {
			"roc_auc": float(roc_auc),
			"pr_auc": float(pr_auc),
			"precision": float(ppv),
			"accuracy": float(acc),
			"precision": float(ppv),
			"f1": float(f1)
		}

		if plot:
			fig, axes = plt.subplots(1, 3, figsize=(16, 4))

			# ROC
			axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
			axes[0].plot([0, 1], [0, 1], "--", lw=1)
			axes[0].set_title(f"ROC — {name}")
			axes[0].set_xlabel("False Positive Rate")
			axes[0].set_ylabel("True Positive Rate")
			axes[0].legend()
			axes[0].grid(True)

			# PR
			axes[1].plot(rec, prec, label=f"AP = {pr_auc:.3f}")
			axes[1].set_title(f"PR — {name}")
			axes[1].set_xlabel("Recall")
			axes[1].set_ylabel("Precision")
			axes[1].legend()
			axes[1].grid(True)

			# Confusion Matrix
			
			cm = confusion_matrix(yt, pt)
			ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=axes[2], colorbar=False)
			axes[2].set_title(f"Confusion Matrix — {name}")

			plt.tight_layout()
			plt.show()

	return report

def plot_running_total(running_total, y_top_pad=0.05, savepath=None):
	"""
	Plots all losses on one chart (raw only), with y-axis starting at 0.
	y_top_pad: extra headroom on top as a fraction of the max (default 5%).
	"""
	keys = [k for k in ["total", "recon", "bce", "supcon"] if k in running_total]
	plt.figure(figsize=(6, 3))

	y_global_max = 0.0
	xs = None
	for k in keys:
		y = np.asarray(running_total[k], dtype=float)
		xs = np.arange(1, len(y) + 1)
		plt.plot(xs, y, label=k)
		y_global_max = max(y_global_max, float(np.max(y)) if len(y) else 0.0)

	# y-axis starts at 0
	top = y_global_max * (1.0 + y_top_pad)
	if top <= 0:  # handle all-zero edge case
		top = 1.0
	plt.ylim(0.0, top)

	# x-axis from epoch 1
	if xs is not None:
		plt.xlim(xs[0], xs[-1])

	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Training Losses per Epoch (raw)")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	if savepath:
		plt.savefig(savepath, dpi=150, bbox_inches="tight")
	plt.show()


def plot_running_total_subplots(running_total, y_top_pad=0.05):
	"""
	One subplot per loss (raw only), each with y-axis starting at 0.
	"""
	keys = [k for k in ["total", "recon", "bce", "supcon"] if k in running_total]
	n = len(keys)
	rows = 2 if n > 2 else 1
	cols = 2 if n > 1 else 1
	fig, axes = plt.subplots(rows, cols, figsize=(6, 3))
	axes = np.array(axes).reshape(-1)

	for ax, k in zip(axes, keys):
		y = np.asarray(running_total[k], dtype=float)
		x = np.arange(1, len(y) + 1)
		ax.plot(x, y)
		ax.set_title(k)
		ax.set_xlabel("Epoch")
		ax.set_ylabel("Loss")
		ax.grid(True, alpha=0.3)

		# y starts at 0, top per-plot
		y_max = float(np.max(y)) if len(y) else 0.0
		top = y_max * (1.0 + y_top_pad)
		if top <= 0:
			top = 1.0
		ax.set_ylim(0.0, top)

		# x from epoch 1
		if len(x):
			ax.set_xlim(x[0], x[-1])

	# hide any unused axes
	for j in range(len(keys), len(axes)):
		axes[j].axis("off")

	plt.tight_layout()
	plt.show()

def _ece(y, p, n_bins=10, strategy="uniform"):
	"""
	Expected Calibration Error with either uniform-width bins or quantile bins.
	strategy : "uniform" -> equal-width bins over [0,1]
			   "quantile" -> ~equal-count bins using quantiles of p
	"""
	y = np.asarray(y, dtype=int)
	p = np.asarray(p, dtype=float)

	if strategy == "quantile":
		# Bin edges at quantiles of the predictions; ties may reduce the
		# effective number of bins (that's fine).
		edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
		edges = np.unique(edges)  # ensure strictly nondecreasing edges
		# Fallback: if everything is identical, edges will have length 1
		if edges.size < 2:
			# all predictions identical -> calibration error is |mean(y) - p0|
			return float(abs(y.mean() - p[0]))
	elif strategy == "uniform":
		edges = np.linspace(0.0, 1.0, n_bins + 1)
	else:
		raise ValueError("strategy must be 'uniform' or 'quantile'")

	# Assign each prediction to a bin: indices in [0, len(edges)]
	idx = np.digitize(p, edges, right=True)

	ece = 0.0
	N = p.size
	# Iterate over actual bins (1..#bins); 0 and len(edges) contain out-of-range by construction
	for b in range(1, len(edges)):
		m = (idx == b)
		if not np.any(m):
			continue
		conf = p[m].mean()		  # mean predicted prob in bin
		acc  = y[m].mean()		  # fraction of positives in bin
		w	= m.sum() / N		  # bin weight
		ece += w * abs(acc - conf)
	return float(ece)

def plot_calibration_curve(probs_to_comp, probs_names, y_true, task_names=["prolonged_stay","mortality","readmission"], n_bins=10, strategy=["uniform","uniform","quantile"], show_scores=True):
	""" probs_to_comp: list of arrays, each shape (N, 3) with probabilities per task.
		probs_names: list of names (len == len(probs_to_comp)).
		y_true: array shape (N, 3) with binary labels.
		One figure per task with 2 subplots: reliability (left) + histograms (right). """
	
	probs_to_comp = [np.asarray(P, float) for P in probs_to_comp]
	y_true = np.asarray(y_true, int)
	T = y_true.shape[1]

	palette = {name: plt.cm.tab10(i % 10) for i, name in enumerate(probs_names)}

	for t in range(T):
		s_task = strategy[t]  # <- use same binning for all methods in this task
		fig, (ax_cal, ax_hist) = plt.subplots(1, 2, figsize=(12, 4))
		ax_cal.plot([0,1],[0,1], "--", color="gray", linewidth=1, label="Perfect")

		for P, name in zip(probs_to_comp, probs_names):
			y_t = y_true[:, t]
			p_t = P[:, t]

			prob_true, prob_pred = calibration_curve(y_t, p_t, n_bins=n_bins, strategy=s_task)

			label = name
			if show_scores:
				ece = _ece(y_t, p_t, n_bins=n_bins, strategy=s_task)
				brier = brier_score_loss(y_t, p_t)
				label = f"{name} (ECE={ece:.3f}, Brier={brier:.3f})"

			color = palette[name]
			ax_cal.plot(prob_pred, prob_true, marker="o", color=color, label=label)
			ax_hist.hist(p_t, bins=n_bins, range=(0,1), alpha=0.45, color=color, label=name, edgecolor="none")

		ax_cal.set_title(f"Calibration — {task_names[t]} (strategy={s_task})")
		ax_cal.set_xlabel("Predicted probability"); ax_cal.set_ylabel("Observed frequency")
		ax_cal.grid(True, alpha=0.3); ax_cal.legend(loc="best")

		ax_hist.set_title(f"Predicted Probabilities — {task_names[t]}")
		ax_hist.set_xlabel("Predicted probability"); ax_hist.set_ylabel("Count")
		ax_hist.grid(True, alpha=0.3); ax_hist.legend(loc="best")

		plt.tight_layout()
		plt.show()