"""
KNN visualization & neighbor summary on task-specific embedding space.

What it does
1) Builds task-specific embeddings for train/test using model.project(z, task_idx)
2) Picks a positive (label=1) test subject the model predicts well
3) Fits KNN on train embeddings and shows a 2D PCA plot:
      - train label=1 in light red
      - train label=0 in light green
      - chosen test subject as a light blue star
4) Prints a short text summary of the neighbors (purity, IDW vote, similarities)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from project.model.prediction import predict_logits, encode_embeddings, predict_proba_knn

# Default task order; adjust if your model uses a different order.
DEFAULT_TASK_TO_IDX = {"mortality": 0,"prolonged_stay": 1,"readmission": 2}


# util helpers

def _device_from_arg(device=None):
    """Pick device from arg or availability."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Subject selection

def _pick_test_index( test_logits, y_test, task_idx, conf_thresh, require_positive, rng=0): 
    """ Pick a test subject to showcase.
        Priority (when require_positive=True) - y=1 & correct & high-confidence """
    N = test_logits.shape[0]
    if N == 0:
        return 0

    rng = np.random.default_rng(rng)
    p = 1.0 / (1.0 + np.exp(-test_logits[:, task_idx]))  # sigmoid
    yhat = (p >= 0.5).astype(int)

    if y_test is not None:
        y_true = y_test[:, task_idx].astype(int)
        correct = (yhat == y_true)
        conf = np.maximum(p, 1 - p)

        if require_positive:
            # positive, correct, high conf
            c1 = np.where((y_true == 1) & (yhat == 1) & (conf >= conf_thresh))[0]
            if c1.size:
                return int(rng.choice(c1))
            # positive & correct
            c2 = np.where((y_true == 1) & (yhat == 1))[0]
            if c2.size:
                return int(rng.choice(c2))
            # positive with highest predicted p
            c3 = np.where(y_true == 1)[0]
            if c3.size:
                return int(c3[np.argmax(p[c3])])

        # any correct, pick highest confidence
        c4 = np.where(correct)[0]
        if c4.size:
            return int(c4[np.argmax(conf[c4])])

    # most confident overall (no labels, or nothing matched above)
    return int(np.argmax(np.maximum(p, 1 - p)))


# KNN + simple neighbor stats

def _idw_vote_probability(neigh_labels, dists, eps=1e-12):
    """ Inverse-distance weighted vote for label=1. """
    w = 1.0 / np.maximum(dists, eps)
    w = w / (w.sum() + eps)
    return float(np.dot(w, neigh_labels.astype(float)))


def visualize_knn_for_subject(model, X_train, y_train, mask_train, X_test, y_test, mask_test, task="mortality", k=15, metric="cosine", conf_thresh=0.8, device=None, rng=0, require_positive=True, task_to_idx=None, draw_neighbor_lines=True):
    """
    Visualize the KNN context for a (positive) test subject on a given task.

    Parameters
    ----------
    model : torch.nn.Module
        Must implement forward(x, lengths) -> (z, logits_list, recon)
        and project(z, task_idx) -> (B, E_task).
    X_train, y_train, mask_train : arrays
        Train set arrays. y_train shape (N, 3).
    X_test, y_test, mask_test : arrays
        Test set arrays. y_test can be None if unavailable.
    task : {"mortality", "prolonged_stay", "readmission"}
        Which task to visualize.
    k : int
        Number of neighbors (will be clipped to len(train)).
    metric : str
        NearestNeighbors metric (e.g., "cosine", "euclidean").
    conf_thresh : float
        Confidence threshold for selecting a "well-predicted" subject.
    device : str or None
        "cuda", "cpu", or None to auto-detect.
    rng : int or np.random.Generator or None
        Random seed/generator for tie-breaking.
    require_positive : bool
        If True, only consider y=1 subjects when selecting.
    task_to_idx : dict or None
        Optional override for task name -> index mapping.
    draw_neighbor_lines : bool
        If True, draw faint lines from test point to each neighbor on the PCA plot.

    Returns
    -------
    info : dict
        Keys include:
          - "task_idx", "test_index"
          - "neighbor_indices", "neighbor_labels", "neighbor_distances"
          - "idw_vote_prob", "label_purity", "cosine_sim_mean", "cosine_sim_top"
    """
    mapping = (task_to_idx or DEFAULT_TASK_TO_IDX).copy()
    if task not in mapping:
        raise ValueError(f"Unknown task '{task}'. Known: {list(mapping.keys())}")
    task_idx = mapping[task]
    dev = _device_from_arg(device)

    # Compute embeddings
    E_tr = encode_embeddings(model, X_train, mask_train, device=str(dev))[task_idx]
    E_te = encode_embeddings(model, X_test,  mask_test,  device=str(dev))[task_idx]

    if E_tr.shape[0] == 0 or E_te.shape[0] == 0:
        raise ValueError("Empty embeddings. Check inputs shapes and mask.")

    # Compute test logits to pick a showcase subject
    logits_te = predict_logits(model, X_test, mask_test, device=str(dev))
    test_index = _pick_test_index(
        logits_te, y_test, task_idx,
        conf_thresh=conf_thresh,
        require_positive=require_positive,
        rng=rng,
    )

    # Fit KNN on train embeddings and query neighbors for the chosen test subject
    k_eff = int(max(1, min(k, len(E_tr))))
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    nn.fit(E_tr)
    dists, idxs = nn.kneighbors(E_te[test_index:test_index + 1], return_distance=True)
    dists, idxs = dists[0], idxs[0]

    # Neighbor labels (0/1) for the task
    neigh_labels = y_train[idxs, task_idx].astype(int)

    # 2D PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    XY_tr = pca.fit_transform(E_tr)
    XY_star = pca.transform(E_te[test_index:test_index + 1])[0]
    XY_neigh = XY_tr[idxs]

    # Plot
    plt.figure(figsize=(8, 7))
    neg = (y_train[:, task_idx].astype(int) == 0)
    pos = ~neg

    # Train clusters
    plt.scatter(
        XY_tr[neg, 0], XY_tr[neg, 1],
        s=14, alpha=0.5, c="lightgreen", edgecolors="none", label="Train label 0"
    )
    plt.scatter(
        XY_tr[pos, 0], XY_tr[pos, 1],
        s=14, alpha=0.5, c="lightcoral", edgecolors="none", label="Train label 1"
    )

    # Chosen test subject
    plt.scatter(
        [XY_star[0]], [XY_star[1]],
        s=180, c="black", edgecolors="k", linewidths=1.0, marker="*",
        label="Test subject"
    )

    # Optional lines from test point to neighbors
    if draw_neighbor_lines:
        for j in range(len(idxs)):
            plt.plot(
                [XY_star[0], XY_neigh[j, 0]],
                [XY_star[1], XY_neigh[j, 1]],
                lw=0.6, alpha=0.25, c="gray"
            )

    plt.title(f"KNN in '{task}' embedding space (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Text summary (kept concise)
    # IDW vote and label purity
    idw_prob = _idw_vote_probability(neigh_labels, dists)
    purity = float(neigh_labels.mean())  # fraction of label=1 among neighbors

    # Cosine similarity derived from distances if metric='cosine'
    cos_mean = cos_top = None
    if metric == "cosine":
        sims = 1.0 - dists
        cos_mean = float(sims.mean())
        cos_top = float(sims[0])

    print("\n=== Neighbor summary ===")
    if y_test is not None:
        true_y = int(y_test[test_index, task_idx])
        p_star = float(1.0 / (1.0 + np.exp(-logits_te[test_index, task_idx])))
        print(f"Test idx: {test_index} | true label={true_y} | predicted p={p_star:.3f}")
    else:
        print(f"Test idx: {test_index} | true label=N/A")

    print(f"k={k_eff} | label-1 purity={purity:.3f} | IDW vote p={idw_prob:.3f}")
    if cos_mean is not None:
        print(f"Cosine similarity: mean={cos_mean:.3f} | top-neighbor={cos_top:.3f}")
    print("Top neighbors (idx, label, dist):")
    for j in range(min(5, len(idxs))):
        print(f"  #{j+1}: idx={idxs[j]} | y={neigh_labels[j]} | dist={dists[j]:.4f}")

    return {
        "task_idx": task_idx,
        "test_index": test_index,
        "neighbor_indices": idxs,
        "neighbor_labels": neigh_labels,
        "neighbor_distances": dists,
        "idw_vote_prob": idw_prob,
        "label_purity": purity,
        "cosine_sim_mean": cos_mean,
        "cosine_sim_top": cos_top,
        # "p_knn_star": p_knn_star,  # if you uncomment the KNN probability block above
    }