import warnings

warnings.filterwarnings("ignore")
# ruff: noqa
import os
import time
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist, pdist, squareform

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import MDS, TSNE
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from sklearn.metrics.cluster import contingency_matrix

from umap import UMAP

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
rng = np.random.default_rng(RANDOM_STATE)

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# =========================================================
# RESULTS LOG  (for DBSCAN experiments summary table)
# =========================================================
dbscan_results = []


# =========================================================
# HELPERS
# =========================================================


def sse(X, labels):
    """Sum of squared errors; ignores noise label -1."""
    total = 0.0
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        mask = labels == lbl
        total += np.sum((X[mask] - X[mask].mean(axis=0)) ** 2)
    return total


def visualize_dbscan(X, labels, stroke=None, title="", filename=""):
    """2-D projection (UMAP preferred, PCA fallback) coloured by DBSCAN label."""
    try:
      X_2d = UMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    except Exception:
      X_2d = PCA(n_components=2).fit_transform(X)

    labels = np.array(labels)
    plt.figure(figsize=(7, 5))
    cmap = plt.cm.get_cmap("tab10")

    for i, lbl in enumerate(np.unique(labels)):
      mask = labels == lbl
      if lbl == -1:
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c="black", s=10, label="noise", alpha=0.6)
      else:
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], color=cmap(i % 10), s=10, label=f"C{lbl}", alpha=0.6)

    if stroke is not None:
      stroke = np.array(stroke)
      plt.scatter(X_2d[stroke == 1, 0], X_2d[stroke == 1, 1], facecolors="none", edgecolors="red", s=40, label="stroke")

      plt.title(title)
      plt.legend(markerscale=2, fontsize=8)
      plt.tight_layout()
      save_path = os.path.join(PLOTS_DIR, filename)
      plt.savefig(save_path, dpi=150)
      plt.close()
      print(f"Saved plot: {filename}")


def run_dbscan_experiment(name, X, eps, min_samples=5, stroke=None):
  """
  Run a single labelled DBSCAN experiment, print diagnostics,
  append to dbscan_results, and save a UMAP/PCA scatter plot.
  Returns the label array.
  """
  print(f"\n=== {name} ===")
  t0 = time.time()
  model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
  labels = model.labels_
  t1 = time.time()

  unique, counts = np.unique(labels, return_counts=True)
  n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
  noise = (labels == -1).sum()

  print(f"eps: {eps}  |  minPts: {min_samples}  |  # clusters: {n_clusters}")
  print("Cluster distribution (%):")
  dist = {}
  for u, c in zip(unique, counts):
    label = "noise" if u == -1 else f"C{u}"
    pct = c / len(labels) * 100
    dist[label] = pct
    print(f"  {label}: {pct:.2f}%")

  runtime = t1 - t0
  print(f"Runtime: {runtime:.3f}s  |  Noise: {noise} ({noise / len(labels) * 100:.2f}%)")

  non_noise = labels != -1
  sil = None
  if len(set(labels[non_noise])) > 1:
    sil = silhouette_score(X[non_noise], labels[non_noise])
    print(f"Silhouette (non-noise): {sil:.4f}")

  stroke_noise = stroke_core = None
  if stroke is not None:
    if noise > 0:
      stroke_noise = stroke[labels == -1].mean()
      print(f"Stroke rate (noise): {stroke_noise:.4f}")
    stroke_core = stroke[labels != -1].mean()
    print(f"Stroke rate (core): {stroke_core:.4f}")

    dbscan_results.append(
      {
        "experiment": name,
        "eps": eps,
        "minPts": min_samples,
        "clusters": n_clusters,
        "noise_pct": noise / len(labels) * 100,
        "runtime": runtime,
        "silhouette": sil,
        "stroke_noise": stroke_noise,
        "stroke_core": stroke_core,
        "distribution": dist,
      }
    )

  visualize_dbscan(X, labels, stroke=stroke, title=name, filename=name.replace(" ", "_").replace("(", "").replace(")", "") + ".png")
  return labels


# =========================================================
# DATA LOADING & PREPROCESSING
# =========================================================
print("\n--- Preprocessing ---")
df_raw = pd.read_csv("healthcare-dataset-stroke-data.csv")

# ── Preprocessing used by the main analysis pipeline ──────────────────────────
df = df_raw.copy()
df.drop(columns=["id"], inplace=True)
df = df[df["gender"] != "Other"].reset_index(drop=True)
df["bmi"] = SimpleImputer(strategy="median").fit_transform(df[["bmi"]])

for col, mapping in [
  ("gender", {"Male": 1, "Female": 0}),
  ("ever_married", {"Yes": 1, "No": 0}),
  ("Residence_type", {"Urban": 1, "Rural": 0}),
]:
  df[col] = df[col].map(mapping)

df = pd.get_dummies(
  df,
  columns=["work_type", "smoking_status"],
  prefix=["work_type", "smoking_status"],
  drop_first=False,
)

stroke_labels = df["stroke"].values
df = df.drop(columns=["stroke"])

cont_cols = ["age", "avg_glucose_level", "bmi"]
df_scaled = df.copy().astype(float)
df_scaled[cont_cols] = StandardScaler().fit_transform(df_scaled[cont_cols])
X = df_scaled.values.astype(float)


# ── Preprocessing used by DBSCAN experiments (full one-hot, smoking score) ────
def preprocess_experiments(df):
  df = df.copy()
  df = df.drop(columns=["id"])
  df = df[df["gender"] != "Other"]
  df["bmi"] = SimpleImputer(strategy="median").fit_transform(df[["bmi"]])
  df = pd.get_dummies(
    df,
    columns=[
      "gender",
      "ever_married",
      "work_type",
      "Residence_type",
      "smoking_status",
    ],
    drop_first=False,
    dtype=int,
  )
  df = df.drop(columns=["gender_Female", "ever_married_No", "Residence_type_Rural"])
  return df


def scale_cols(df, cols):
  df = df.copy()
  df[cols] = StandardScaler().fit_transform(df[cols])
  return df


data_exp = preprocess_experiments(df_raw)
data_exp_sc = scale_cols(data_exp, cont_cols)
X_exp = data_exp_sc.drop(columns=["stroke"]).values
y_exp = data_exp_sc["stroke"].values


# =========================================================
# K-MEANS CLUSTERING
# =========================================================
print("\n--- K-Means Clustering ---")

sse_list, sil_list, km_models = [], [], {}

for k in range(1, 11):
  km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
  km.fit(X)
  km_models[k] = km
  sse_list.append(km.inertia_)
  sil_list.append(silhouette_score(X, km.labels_) if k > 1 else float("nan"))

print("\n  k  |    SSE      | Silhouette")
print("  ---+-------------+-----------")
for i, k in enumerate(range(1, 11)):
  s = f"{sil_list[i]:.4f}" if not np.isnan(sil_list[i]) else "  N/A "
  print(f"  {k:2d} | {sse_list[i]:11.2f} | {s}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, 11), sse_list, marker="o", color="steelblue")
axes[0].axvline(4, color="red", linestyle="--", label="chosen k=4")
axes[0].set_xlabel("k")
axes[0].set_ylabel("SSE")
axes[0].set_title("K-Means SSE Elbow Curve")
axes[0].legend()

axes[1].plot(range(2, 11), sil_list[1:], marker="s", color="darkorange")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette")
axes[1].set_title("K-Means Silhouette Scores")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "kmeans_elbow_silhouette.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: kmeans_elbow_silhouette.png")

BEST_K = 4
km_best = km_models[BEST_K]
km_labels = km_best.labels_
km_centers = km_best.cluster_centers_

print(f"\nChosen k = {BEST_K}")
print(f"  SSE       : {km_best.inertia_:.2f}")
print(f"  Silhouette: {silhouette_score(X, km_labels):.4f}")
print(f"  Cluster sizes: {dict(zip(*np.unique(km_labels, return_counts=True)))}")

df_viz = df.copy()
df_viz["km_label"] = km_labels
print("\nPer-cluster means (original scale):")
print(df_viz.groupby("km_label")[cont_cols + ["hypertension", "heart_disease"]].mean().round(3))

# Per-sample silhouette plot
sil_vals = silhouette_samples(X, km_labels)
fig, ax = plt.subplots(figsize=(10, 6))
cmap4 = plt.cm.get_cmap("tab10")
y_lower = 10

for i in range(BEST_K):
  cluster_sil = np.sort(sil_vals[km_labels == i])
  y_upper = y_lower + cluster_sil.shape[0]
  ax.fill_betweenx(
    np.arange(y_lower, y_upper),
    0,
    cluster_sil,
    facecolor=cmap4(i / BEST_K),
    edgecolor=cmap4(i / BEST_K),
    alpha=0.7,
  )
  ax.text(-0.05, y_lower + 0.5 * cluster_sil.shape[0], str(i), fontsize=10)
  y_lower = y_upper + 10

ax.axvline(silhouette_score(X, km_labels), color="red", linestyle="--", label="Average")
ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.set_title(f"Per-Sample Silhouette – K-Means (k={BEST_K})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "silhouette_kmeans.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: silhouette_kmeans.png")

# PCA scatter across k=2–6  (from Kfeliz notebook)
print("\n  PCA scatter for k=2..6 …")
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X)

k_range_pca = [2, 3, 4, 5, 6]
kmeans_pca_results = []
fig, axes = plt.subplots(1, len(k_range_pca), figsize=(25, 5))

for i, k in enumerate(k_range_pca):
  km_tmp = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
  lbl_tmp = km_tmp.fit_predict(X)
  axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=lbl_tmp, s=10, alpha=0.6, cmap="tab10")
  axes[i].set_title(f"K-Means k={k}")
  axes[i].set_xlabel("PCA 1")
  axes[i].set_ylabel("PCA 2")
  cluster_pcts = (pd.Series(lbl_tmp).value_counts(normalize=True) * 100).round(2)
  kmeans_pca_results.append(
    {
      "k": k,
      "SSE": km_tmp.inertia_,
      "Silhouette": silhouette_score(X, lbl_tmp),
      "Cluster %": cluster_pcts.to_dict(),
    }
  )

plt.suptitle("K-Means PCA Scatter (k=2 to 6)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "kmeans_pca_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: kmeans_pca_scatter.png")
print(pd.DataFrame(kmeans_pca_results)[["k", "SSE", "Silhouette"]].to_string(index=False))


# =========================================================
# HIERARCHICAL CLUSTERING
# =========================================================
print("\n--- Hierarchical Clustering ---")

sample_idx = rng.choice(len(X), size=400, replace=False)
X_sub = X[sample_idx]

linkage_methods = ["ward", "complete", "average", "single"]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.ravel()

for i, method in enumerate(linkage_methods):
  Z = linkage(X_sub, method=method)
  dendrogram(
    Z,
    ax=axes[i],
    truncate_mode="lastp",
    p=25,
    leaf_rotation=90,
    leaf_font_size=7,
    color_threshold="default",
  )
  axes[i].set_title(f"Dendrogram – {method.capitalize()} Linkage", fontsize=12)
  axes[i].set_xlabel("Cluster / data index")
  axes[i].set_ylabel("Distance")

plt.suptitle("Hierarchical Clustering Dendrograms", fontsize=16)
plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "hierarchical_dendrograms.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("Saved: hierarchical_dendrograms.png")

hc_labels_dict = {}
print(f"\nAgglomerative Clustering (k={BEST_K}) on full dataset:")
print(f"  {'Method':<12} | {'SSE':>10} | {'Silhouette':>10} | Cluster sizes")
print(f"  {'-' * 12}-+-{'-' * 10}-+-{'-' * 10}-+------------------")

for method in linkage_methods:
  agg = AgglomerativeClustering(n_clusters=BEST_K, linkage=method)
  lbl = agg.fit_predict(X)
  hc_labels_dict[method] = lbl
  sizes = dict(zip(*np.unique(lbl, return_counts=True)))
  print(f"  {method:<12} | {sse(X, lbl):>10.2f} | {silhouette_score(X, lbl):>10.4f} | {sizes}")


# =========================================================
# DBSCAN  – PART 1: Grid Search on Main Dataset
# =========================================================
print("\n--- DBSCAN Grid Search (main dataset) ---")

K_NN = 5
nbrs = NearestNeighbors(n_neighbors=K_NN).fit(X)
distances, _ = nbrs.kneighbors(X)
k_dists = np.sort(distances[:, K_NN - 1])[::-1]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(k_dists, color="teal")
ax.axhline(y=1.5, color="red", linestyle="--", label="eps ≈ 1.5")
ax.set_xlabel("Data points (sorted)")
ax.set_ylabel(f"{K_NN}-NN Distance")
ax.set_title("k-NN Distance Plot – Estimating DBSCAN Epsilon")
ax.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(PLOTS_DIR, "dbscan_knn_distance.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print("Saved: dbscan_knn_distance.png")

eps_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
min_pts_vals = [3, 5, 10, 15]
best_cfg, best_sil = None, -np.inf

print("\n  eps  | min_pts | n_clusters | noise %  | Silhouette")
print("  -----+---------+------------+----------+-----------")

for eps in eps_values:
  for mpts in min_pts_vals:
    lbl = DBSCAN(eps=eps, min_samples=mpts).fit_predict(X)
    n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
    noise = (lbl == -1).sum() / len(lbl) * 100
    if n_cl >= 2:
      sil = silhouette_score(X, lbl)
      if sil > best_sil:
        best_sil = sil
        best_cfg = (eps, mpts)
    else:
      sil = float("nan")
    s = f"{sil:.4f}" if not np.isnan(sil) else "   N/A"
    print(f"  {eps:>4.1f} | {mpts:>7} | {n_cl:>10} | {noise:>7.1f}% | {s}")

print(f"\nBest config: eps={best_cfg[0]}, min_samples={best_cfg[1]}")

db_best = DBSCAN(eps=best_cfg[0], min_samples=best_cfg[1])
db_labels = db_best.fit_predict(X)
n_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
noise_cnt = (db_labels == -1).sum()

print(f"Clusters: {n_db}")
print(f"Noise: {noise_cnt} ({noise_cnt / len(X) * 100:.1f}%)")
print(f"Silhouette (non-noise): {silhouette_score(X, db_labels):.4f}")
print(f"Cluster sizes: {dict(zip(*np.unique(db_labels, return_counts=True)))}")


# =========================================================
# DBSCAN  – PART 2: Structured Experiments (Experiments.py)
# =========================================================
print("\n--- DBSCAN Structured Experiments ---")

# Experiment 1 – Extreme anomaly detection
labels_e1 = run_dbscan_experiment("Experiment 1 (eps=1.75)", X_exp, eps=1.75, stroke=y_exp)

# Experiment 2 – Stroke-only clustering
stroke_only = data_exp_sc[data_exp_sc["stroke"] == 1].drop(columns=["stroke"])
labels_e2 = run_dbscan_experiment("Experiment 2 (stroke only)", stroke_only.values, eps=1.3)

# Experiment 3 – Age-group quantile clustering
ex3 = data_exp_sc.copy()
ex3["age_group"] = pd.qcut(ex3["age"], q=3, labels=["young", "middle", "old"])
for group_name in ["young", "middle", "old"]:
  mask = ex3["age_group"] == group_name
  X_group = ex3[mask].drop(columns=["age_group", "stroke"]).values
  y_group = ex3[mask]["stroke"].values
  run_dbscan_experiment(f"Experiment 3 ({group_name})", X_group, eps=1.4, stroke=y_group)

# Experiment 4 – Feature reduction + composite smoking score
ex4 = data_exp_sc.drop(columns=["work_type_Govt_job", "work_type_Never_worked", "work_type_Private"])
ex4 = ex4.copy()
ex4["smoking_score"] = (
    0.0 * ex4["smoking_status_never smoked"]
    + 0.5 * ex4["smoking_status_formerly smoked"]
    + 0.5 * ex4["smoking_status_Unknown"]
    + 1.0 * ex4["smoking_status_smokes"]
)
ex4 = ex4.drop(
    columns=[
        "smoking_status_never smoked",
        "smoking_status_formerly smoked",
        "smoking_status_smokes",
        "smoking_status_Unknown",
        "stroke",
    ]
)
labels_e4 = run_dbscan_experiment("Experiment 4 (feature reduction)", ex4.values, eps=1.3, stroke=y_exp)

# Experiment 5 – Weighted hypertension & heart disease
ex5 = data_exp_sc.drop(columns=["stroke"]).copy()
ex5[["hypertension", "heart_disease"]] *= 3
labels_e5 = run_dbscan_experiment("Experiment 5 (weighted features)", ex5.values, eps=1.6, stroke=y_exp)

# DBSCAN experiments summary
print("\n================ DBSCAN EXPERIMENT SUMMARY ================\n")
summary_df = pd.DataFrame(dbscan_results)
print(
  summary_df[
    [
      "experiment",
      "eps",
      "minPts",
      "clusters",
      "noise_pct",
      "runtime",
      "silhouette",
      "stroke_core",
      "stroke_noise",
    ]
  ].to_string(index=False)
)


# =========================================================
# ANOMALY DETECTION
# =========================================================
print("\n--- Anomaly Detection ---")

THRES_HLD = 97

# K-Means: distance to nearest cluster centre
km_dist = np.min(cdist(X, km_centers), axis=1)
km_thresh = np.percentile(km_dist, THRES_HLD)
km_out_idx = np.where(km_dist > km_thresh)[0]
not_km_idx = np.where(km_dist <= km_thresh)[0]

print(f"\nK-Means Anomaly  (score = dist to nearest center)")
print(f"{THRES_HLD}th-pct threshold: {km_thresh:.4f}")
print(f"Outliers detected: {len(km_out_idx)}")
print(f"Stroke % outliers: {stroke_labels[km_out_idx].mean() * 100:.1f}%")
print(f"Stroke % inliers: {stroke_labels[not_km_idx].mean() * 100:.1f}%")

df_viz["km_anomaly_score"] = km_dist
df_viz["is_km_outlier"] = (km_dist > km_thresh).astype(int)
print("\n  K-Means outlier vs. inlier continuous means:")
print(df_viz.groupby("is_km_outlier")[cont_cols].mean().round(2))

# Hierarchical Ward: distance to own centroid
ward_lbl = hc_labels_dict["ward"]
ward_cent = np.array([X[ward_lbl == i].mean(0) for i in range(BEST_K)])
ward_dist = np.array([np.linalg.norm(X[i] - ward_cent[ward_lbl[i]]) for i in range(len(X))])
ward_thresh = np.percentile(ward_dist, THRES_HLD)
ward_out_idx = np.where(ward_dist > ward_thresh)[0]
not_ward_idx = np.where(ward_dist <= ward_thresh)[0]

print(f"\nHierarchical (Ward) Anomaly  (score = dist to own centroid)")
print(f"{THRES_HLD}th-pct threshold: {ward_thresh:.4f}")
print(f"Outliers detected: {len(ward_out_idx)}")
print(f"Stroke % outliers: {stroke_labels[ward_out_idx].mean() * 100:.1f}%")
print(f"Stroke % inliers: {stroke_labels[not_ward_idx].mean() * 100:.1f}%")

# DBSCAN: noise == outlier
db_out_idx = np.where(db_labels == -1)[0]
db_core_idx = np.where(db_labels != -1)[0]

print(f"\nDBSCAN Anomaly  (noise label = -1)")
print(f"Noise points (outliers): {len(db_out_idx)}")
if len(db_out_idx) > 0:
  print(f"Stroke % outliers: {stroke_labels[db_out_idx].mean() * 100:.1f}%")
print(f"Stroke % inliers: {stroke_labels[db_core_idx].mean() * 100:.1f}%")

# Overlap
km_set = set(km_out_idx)
ward_set = set(ward_out_idx)
db_set = set(db_out_idx)
all3 = km_set & ward_set & db_set
km_ward = km_set & ward_set

print(f"\nOutlier overlap:")
print(f"K-Means ∩ Ward: {len(km_ward)} points")
print(f"All three methods: {len(all3)} points")

if len(all3) > 0:
  all3_arr = np.array(list(all3))
  print(f"Stroke rate (all-3): {stroke_labels[all3_arr].mean() * 100:.1f}%")
  print(f"All-3 outlier continuous feature stats:")
  print(df.iloc[all3_arr][cont_cols].describe().round(2))

# Anomaly score distributions plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(km_dist, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
axes[0].axvline(km_thresh, color="red", linestyle="--", label=f"97th pct={km_thresh:.2f}")
axes[0].set_title("K-Means: Dist to Nearest Center")
axes[0].set_xlabel("Anomaly Score")
axes[0].legend()

axes[1].hist(ward_dist, bins=60, color="darkorange", edgecolor="white", alpha=0.8)
axes[1].axvline(ward_thresh, color="red", linestyle="--", label=f"97th pct={ward_thresh:.2f}")
axes[1].set_title("Hier (Ward): Dist to Own Centroid")
axes[1].set_xlabel("Anomaly Score")
axes[1].legend()

noise_mask = db_labels == -1
axes[2].bar(
  ["Non-noise", "Noise (outlier)"],
  [(~noise_mask).sum(), noise_mask.sum()],
  color=["teal", "crimson"],
)
axes[2].set_title("DBSCAN: Noise vs Core/Border")
axes[2].set_ylabel("Count")

plt.suptitle("Anomaly Score Distributions", fontsize=14)
plt.tight_layout()
plt.savefig(
  os.path.join(PLOTS_DIR, "anomaly_score_distributions.png"),
  dpi=150,
  bbox_inches="tight",
)
plt.close()
print("\nSaved: anomaly_score_distributions.png")


# =========================================================
# INTERNAL INDICES
# =========================================================
print("\n--- Internal Indices ---")
print(f"\n  {'Method':<24} | {'SSE':>12} | {'Silhouette':>10}")
print(f"  {'-' * 24}-+-{'-' * 12}-+-{'-' * 10}")

all_methods = [("K-Means k=4", km_labels)]
for m in linkage_methods:
  all_methods.append((f"Hier-{m} k=4", hc_labels_dict[m]))

nonnoise = db_labels != -1
db_sil = (
  silhouette_score(X[nonnoise], db_labels[nonnoise])
  if len(set(db_labels[nonnoise])) >= 2
  else float("nan")
)
all_methods.append(("DBSCAN (grid best)", db_labels))

for name, lbl in all_methods:
  s_sse = sse(X, lbl)
  s_sil = db_sil if name.startswith("DBSCAN") else silhouette_score(X, lbl)
  sil_str = f"{s_sil:.4f}" if not np.isnan(s_sil) else "  N/A "
  print(f"  {name:<24} | {s_sse:>12.2f} | {sil_str:>10}")

# Proximity / incidence heatmap
HM_N = 300
hm_idx = rng.choice(len(X), HM_N, replace=False)
X_hm = X[hm_idx]
lbl_hm = km_labels[hm_idx]
order = np.argsort(lbl_hm)
X_hm = X_hm[order]
lbl_hm = lbl_hm[order]

dist_mat = squareform(pdist(X_hm, metric="euclidean"))
inc_mat = (lbl_hm[:, None] == lbl_hm[None, :]).astype(float)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
im0 = axes[0].imshow(dist_mat, cmap="viridis_r", aspect="auto")
axes[0].set_title("Distance (Proximity) Matrix")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(inc_mat, cmap="Blues", aspect="auto")
axes[1].set_title("Incidence Matrix (same cluster = 1)")
plt.colorbar(im1, ax=axes[1])

plt.suptitle(f"K-Means k={BEST_K}: Proximity vs Incidence (n={HM_N})", fontsize=13)
plt.tight_layout()
plt.savefig(
  os.path.join(PLOTS_DIR, "proximity_incidence_heatmap.png"),
  dpi=150,
  bbox_inches="tight",
)
plt.close()
print("Saved: proximity_incidence_heatmap.png")

# Distance–incidence Pearson correlation  (from Kfeliz notebook)
# Negative correlation expected: same-cluster pairs (incidence=1) should have lower distance
from sklearn.metrics import pairwise_distances as _pw_dist

_dist_full = _pw_dist(X)
_inc_full = (km_labels[:, None] == km_labels[None, :]).astype(int)
_corr = np.corrcoef(_dist_full.flatten(), _inc_full.flatten())[0, 1]
print(f"\n  Pearson correlation (distance vs cluster membership): {_corr:.4f}")
print("  (Negative = same-cluster pairs tend to be closer, as expected)")


# =========================================================
# RELATIVE INDICES
# =========================================================
print("\n--- Relative Indices ---")


def relative_metrics(la, lb, na, nb):
  ari = adjusted_rand_score(la, lb)
  nmi = normalized_mutual_info_score(la, lb)
  ami = adjusted_mutual_info_score(la, lb)
  print(f"\n  {na}  vs  {nb}")
  print(f"    ARI={ari:.4f}  NMI={nmi:.4f}  AMI={ami:.4f}")
  print(f"    SSE: {sse(X, la):.2f}  vs  {sse(X, lb):.2f}")


relative_metrics(km_labels, hc_labels_dict["ward"], "K-Means k=4", "Hier-Ward k=4")
km_seed2 = KMeans(n_clusters=BEST_K, n_init=10, random_state=7).fit(X)
relative_metrics(km_labels, km_seed2.labels_, "K-Means seed=42", "K-Means seed=7")
relative_metrics(
  hc_labels_dict["ward"],
  hc_labels_dict["complete"],
  "Hier-Ward k=4",
  "Hier-Complete k=4",
)
relative_metrics(
  hc_labels_dict["average"],
  hc_labels_dict["single"],
  "Hier-Average k=4",
  "Hier-Single k=4",
)


# =========================================================
# EXTERNAL INDICES
# =========================================================
print("\n--- External Indices ---")


def external_metrics(lbl_pred, method_name):
  h, c, v = (
    homogeneity_score(stroke_labels, lbl_pred),
    completeness_score(stroke_labels, lbl_pred),
    v_measure_score(stroke_labels, lbl_pred),
  )
  ari = adjusted_rand_score(stroke_labels, lbl_pred)
  nmi = normalized_mutual_info_score(stroke_labels, lbl_pred)
  ami = adjusted_mutual_info_score(stroke_labels, lbl_pred)
  print(f"\n  {method_name}")
  print(f"    Homogeneity={h:.4f}  Completeness={c:.4f}  V-Measure={v:.4f}")
  print(f"    ARI={ari:.4f}  NMI={nmi:.4f}  AMI={ami:.4f}")


external_metrics(km_labels, f"K-Means k={BEST_K}")
external_metrics(hc_labels_dict["ward"], f"Hierarchical Ward k={BEST_K}")
external_metrics(hc_labels_dict["complete"], f"Hierarchical Complete k={BEST_K}")
external_metrics(db_labels, "DBSCAN (grid best)")

# Contingency matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (lbl, name) in zip(
  axes,
  [
    (km_labels, f"K-Means k={BEST_K}"),
    (hc_labels_dict["ward"], f"Hier-Ward k={BEST_K}"),
    (db_labels, "DBSCAN"),
  ],
):
  cm = contingency_matrix(stroke_labels, lbl)
  sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    ax=ax,
    xticklabels=[f"C{i}" for i in range(cm.shape[1])],
    yticklabels=["No Stroke", "Stroke"],
  )
  ax.set_title(f"Contingency: {name}")
  ax.set_xlabel("Cluster")
  ax.set_ylabel("Stroke Label")

plt.suptitle("Contingency Matrices  (Clustering vs Stroke Label)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "contingency_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: contingency_matrices.png")


# =========================================================
# VISUALIZATION  –  MDS / t-SNE / UMAP  (n=500 subsample)
# =========================================================
print("\n--- Visualization ---")

VIS_N = 500
vis_idx = rng.choice(len(X), VIS_N, replace=False)
X_vis = X[vis_idx]

print("  Running MDS …")
X_mds = MDS(n_components=2, random_state=RANDOM_STATE, dissimilarity="euclidean").fit_transform(X_vis)

print("  Running t-SNE …")
X_tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30).fit_transform(X_vis)

print("  Running UMAP …")
X_umap = UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=20, min_dist=0.1).fit_transform(X_vis)

lbl_vis = {
  "K-Means k=4": km_labels[vis_idx],
  "Hier-Ward k=4": hc_labels_dict["ward"][vis_idx],
  "DBSCAN": db_labels[vis_idx],
  "Stroke Label": stroke_labels[vis_idx],
}
projections = [("MDS", X_mds), ("t-SNE", X_tsne), ("UMAP", X_umap)]
label_items = list(lbl_vis.items())
cmap_vis = plt.cm.get_cmap("tab10")

fig, axes = plt.subplots(3, 4, figsize=(22, 14))

for row, (proj_name, X_proj) in enumerate(projections):
  for col, (label_name, labels) in enumerate(label_items):
    ax = axes[row][col]
    uniq = np.unique(labels)
    for j, lv in enumerate(uniq):
      mask = labels == lv
      color = "grey" if lv == -1 else cmap_vis(j / max(len(uniq), 1))
      ax.scatter(
        X_proj[mask, 0],
        X_proj[mask, 1],
        c=[color],
        s=8,
        alpha=0.6,
        label=("Noise" if lv == -1 else str(lv)),
      )
    ax.set_title(f"{proj_name} – {label_name}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
      ax.set_ylabel(proj_name, fontsize=11)
      if row == 0:
        ax.legend(markerscale=2, fontsize=7, loc="upper right")

plt.suptitle("Cluster Visualizations: MDS | t-SNE | UMAP  (n=500)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cluster_visualizations.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cluster_visualizations.png")

print("\n=== Pipeline complete. All plots saved to:", PLOTS_DIR, "===")
