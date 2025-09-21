import numpy as np

import shap
import matplotlib.pyplot as plt

import torch, copy
from torch.utils.data import DataLoader
from model.dataset import TimeSeriesDataset, BalancedPositivesPerTaskSampler 
import torch.nn as nn

class TorchSeqWrapper(nn.Module):
    """
    MultiTaskSeqGRUAE so GradientExplainer can call it as f(X)->p(task).
    - model(x, lengths) -> z, logits(list of 3), x_hat
    - keep a "current mask" to build lengths for whatever
      batch SHAP passes in.
    """
    def __init__(self, model, task_idx, device="cpu"):
        super().__init__()
        self.task_idx = int(task_idx)
        self.device = torch.device(device)  # <— hard-force SHAP on CPU
        self.model = copy.deepcopy(model).to(self.device).eval()  # <— always clone to CPU

    def forward(self, X_withmask):
        """ X_withMask is one input, last channel is the mask 
            Returns torch tensor with probabilities for the chosen task """
        X_withmask = X_withmask.to(self.device).requires_grad_(True)
   
        X = X_withmask[..., :-1] # (B, T, D)
        M = X_withmask[..., -1].detach()  # (B, T)

        # lengths per *current* mini-batch (size B)
        lengths = M.sum(dim=1).clamp(min=1, max=M.size(1)).long()
        
        # models forward -> z, logits, x_hat  
        _prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            z, logits, _ = self.model(X, lengths)
        finally:
            torch.backends.cudnn.enabled = _prev

        # select chosen task and sigmoid -> prob
        return torch.sigmoid(logits[self.task_idx]).unsqueeze(-1)

def stack_X_and_M(X, M):
    """Pack mask as last channel: (B,T,D) + (B,T,1) -> (B,T,D+1)"""
    M_last = M[..., None].astype(X.dtype) # (B,T,1)
    return np.concatenate([X, M_last], axis=-1) # (B,T,D+1)

def to_torch(x_np):
    return torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

def get_batches(X, y, mask, batch_size=64, p_per_task=6, n_batches=2, seed=42):
    """ Returns (X_batch, M_batch) stacked from n balanced batches our sampler """
    ds = TimeSeriesDataset(X, y, mask)
    sampler = BalancedPositivesPerTaskSampler(y, batch_size=batch_size, p_per_task=p_per_task, seed=seed)
    loader = DataLoader(ds, batch_sampler=sampler)

    Xs, Ms = [], []
    for i, (xb, _, mb, _) in enumerate(loader):
        Xs.append(xb.numpy()); Ms.append(mb.numpy())
        if i + 1 >= n_batches:
            break
    Xb = np.concatenate(Xs, axis=0)
    Mb = np.concatenate(Ms, axis=0)
    keep_bg = (Mb.sum(axis=1) > 0) # drop rows with zero valid timesteps (safer than pretending length=1)
    Xb, Mb = Xb[keep_bg], Mb[keep_bg]
    return Xb, Mb

def feature_importance(model, task_idx, X_train, y_train, mask_train, X_test, y_test, mask_test, fnames, batch_size=64, p_per_task=6, n_batches_train=10, n_batches_test=10):
    X_bg, M_bg = get_batches(X_train, y_train, mask=mask_train, batch_size=batch_size, p_per_task=p_per_task, n_batches=n_batches_train)
    X_ev, M_ev = get_batches(X_test, y_test, mask=mask_test, batch_size=batch_size, p_per_task=p_per_task, n_batches=n_batches_test)

    # pack mask as last channel
    Xbg_cat = stack_X_and_M(X_bg, M_bg) # (B,T,D+1)
    Xev_cat = stack_X_and_M(X_ev, M_ev)

    # build wrapper
    wrapped = TorchSeqWrapper(model, task_idx=task_idx, device="cpu")
    # build explainer
    explainer = shap.GradientExplainer(wrapped, to_torch(Xbg_cat))
    sv = explainer(to_torch(Xev_cat), nsamples=500)
    vals = np.asarray(sv.values)[..., 0] 

    # drop mask channel (last feature)
    vals_feat = vals[..., :-1] # (Ne, T, F)
    X_feat = Xev_cat[..., :-1] # (Ne, T, F)
    Ne, T, F = vals_feat.shape

    # global mean |SHAP| per feature averaged over all patients and all time steps
    mean_abs_feat = np.abs(vals_feat).mean(axis=(0, 1)).reshape(-1)  # (F,)
    order = np.argsort(mean_abs_feat)[::-1]
    ranked = [(fnames[int(i)], float(mean_abs_feat[int(i)])) for i in order]

    # Bar plot (top-k) - which features matter most overall
    topk = min(25, F); idx = order[:topk]
    plt.figure(figsize=(6, 5))
    plt.barh([fnames[int(i)] for i in idx[::-1]], mean_abs_feat[idx][::-1])
    plt.title(f"SeqModel Mean |SHAP| by Feature (task {task_idx})")
    plt.tight_layout(); plt.show()

    # mean absolute SHAP per (time step, feature), averaged over patients
    mean_abs_ft = np.abs(vals_feat).mean(axis=0)  # (T, F)
    # Feature×time heatmap - when (which time steps) each feature matters (to spot temporal patterns)
    plt.figure(figsize=(10, 6))
    plt.imshow(mean_abs_ft[:, idx].T, aspect="auto", interpolation="nearest")
    plt.yticks(range(topk), [fnames[int(i)] for i in idx])
    plt.xlabel("Time step"); plt.title(f"Mean |SHAP| Heatmap (feature×time, task {task_idx})")
    plt.colorbar(label="|SHAP|"); plt.tight_layout(); plt.show()

    # Beeswarm-like (flatten time)
    # How each (feature@time) pushes predictions up/down across patients
    def _base_vals_or_none(sv, Ne):
        b = getattr(sv, "base_values", None) or getattr(sv, "expected_value", None)
        if b is None:
            return None
        b = np.asarray(b)
        if b.ndim == 0:
            return np.full(Ne, float(b))
        return b.reshape(Ne, -1).mean(axis=1)

    bv = _base_vals_or_none(sv, Ne)

    flat_vals = vals_feat.reshape(Ne, -1)                # (Ne, T*F)
    flat_data = X_feat.reshape(Ne, -1)                   # (Ne, T*F)
    names_flat = [f"{f}@t{t}" for t in range(T) for f in fnames]

    kwargs = dict(values=flat_vals, data=flat_data, feature_names=names_flat)
    if bv is not None:
        kwargs["base_values"] = bv

    exp_flat = shap.Explanation(**kwargs)
    shap.plots.beeswarm(exp_flat, max_display=30, show=False)
    plt.title(f"Beeswarm (top 30 feat@time, task {task_idx})")
    plt.tight_layout(); plt.show()

    return ranked

def feature_importance_pretty(
    model, task_idx, X_train, y_train, mask_train, X_test, y_test, mask_test, fnames,
    batch_size=64, p_per_task=6, n_batches_train=10, n_batches_test=10,
    topk=20,
    heatmap_mode="relative",  # "relative" (row 0-1) or "absolute"
    select_for_heatmap="importance",  # "importance" or "time_var"
    time_agg="sum"  # when beeswarm="features": "sum" | "mean" | "l2"
):
    """
    - heatmap_mode - colors scale in the feature×time heatmap
        * "relative" (row-wise 0–1 scaling): highlights when each feature matters most - that is for revealing temporal peaks and patterns (regardless of its overall size).
        * "absolute": shows how strong features are in relations to each other.
    - select_for_heatmap - chooses features for the heatmap by patterns
        * "importance": pick the features with the highest overall mean |SHAP|
        * "time_var": pick the features whose importance changes most over time (highest variance across time).
    - time_agg - how to combine SHAP values across time for each feature in beeswarm
        * "sum": sums SHAP over time - total directional effect across time
        * "mean": averages SHAP over time - “typical per-step” effect.
    Returns a dict with the processed arrays plus the ranked list.
    """

    # Background/Eval sets and SHAP
    X_bg, M_bg = get_batches(X_train, y_train, mask_train, batch_size, p_per_task, n_batches_train)
    X_ev, M_ev = get_batches(X_test,  y_test,  mask_test,  batch_size, p_per_task, n_batches_test)

    Xbg_cat = stack_X_and_M(X_bg, M_bg)
    Xev_cat = stack_X_and_M(X_ev, M_ev)

    wrapped = TorchSeqWrapper(model, task_idx=task_idx, device="cpu")
    explainer = shap.GradientExplainer(wrapped, to_torch(Xbg_cat))
    sv = explainer(to_torch(Xev_cat), nsamples=300)

    # Normalize sv.values to (Ne, T, F_all)
    vals = np.asarray(sv.values)[..., 0]

    vals_feat = vals[..., :-1]       # (Ne, T, F)  drop mask
    X_feat   = Xev_cat[..., :-1]     # (Ne, T, F)

    # global importance - mean |SHAP| over samples & time
    mean_abs_feat = np.abs(vals_feat).mean(axis=(0, 1))  # (F,)
    order_imp = np.argsort(mean_abs_feat)[::-1]
    ranked = [(fnames[int(i)], float(mean_abs_feat[int(i)])) for i in order_imp]

    # time importance matrix - mean |SHAP| over samples and times
    mean_abs_ft = np.abs(vals_feat).mean(axis=0)  # (T, F)

    # choose features for heatmap
    if select_for_heatmap == "importance":
        idx = order_imp[:topk]
        subtitle = "Top features by overall importance"
    elif select_for_heatmap == "time_var":
        time_var = mean_abs_ft.var(axis=0)        # variance across time per feature
        order_var = np.argsort(time_var)[::-1]
        idx = order_var[:topk]
        subtitle = "Top features by time-variance of importance"
    else:
        raise ValueError("select_for_heatmap must be 'importance' or 'time_var'")

    # Bar chart (global importance)
    plt.figure(figsize=(6, 5))
    labels = [fnames[int(i)] for i in order_imp[:topk]][::-1]
    vals_bar = mean_abs_feat[order_imp[:topk]][::-1]
    plt.barh(labels, vals_bar)
    plt.title(f"Global mean |SHAP| by feature (task {task_idx})")
    plt.tight_layout(); plt.show()

    # Heatmap
    H = mean_abs_ft[:, idx].T  # (topk, T) rows=features, cols=time
    if heatmap_mode == "relative":
        # row-wise 0-1 scaling to reveal *patterns* not magnitudes
        denom = H.max(axis=1, keepdims=True) + 1e-12
        H_disp = H / denom
        cbar_label = "relative |SHAP| (per feature)"
    elif heatmap_mode == "absolute":
        # percentile clip to avoid being dominated by a few features
        vmin = np.percentile(H, 5); vmax = np.percentile(H, 95)
        H_disp = np.clip(H, vmin, vmax)
        cbar_label = "|SHAP| (clipped)"
    else:
        raise ValueError("heatmap_mode must be 'relative' or 'absolute'")

    plt.figure(figsize=(10, 6))
    plt.imshow(H_disp, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(idx)), [fnames[int(i)] for i in idx])
    plt.xlabel("Time step")
    plt.title(f"Mean |SHAP| Heatmap (task {task_idx}) — {subtitle}\n"
              f"mode: {heatmap_mode}")
    plt.colorbar(label=cbar_label)
    plt.tight_layout(); plt.show()

    # Beeswarm
    # Aggregate SHAP over time per sample&feature → direction preserved
    if time_agg == "sum":
        vals_bee = vals_feat.sum(axis=1)     # (Ne, F)
    elif time_agg == "mean":
        vals_bee = vals_feat.mean(axis=1)    # (Ne, F)
    else:
        raise ValueError("time_agg must be 'sum'|'mean'|'l2'")

    # feature value for color: use mean over time of the input feature
    data_bee = X_feat.mean(axis=1)           # (Ne, F)

    # restrict to the same topk (by overall importance) for readability
    keep_cols = order_imp[:topk]
    vals_bee_k = vals_bee[:, keep_cols]
    data_bee_k = data_bee[:, keep_cols]
    names_k = [fnames[int(i)] for i in keep_cols]

    exp = shap.Explanation(values=vals_bee_k, data=data_bee_k, feature_names=names_k)
    shap.plots.beeswarm(exp, max_display=topk, show=False)
    plt.title(f"Beeswarm (features, time-agg={time_agg}, task {task_idx})")
    plt.tight_layout(); plt.show()
    
    return {
        "ranked": ranked,
        "mean_abs_feat": mean_abs_feat,
        "mean_abs_ft": mean_abs_ft,
        "order_imp": order_imp,
        "idx_heatmap": idx
    }