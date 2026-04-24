import warnings
warnings.filterwarnings("ignore")
 
import numpy as np # noqa: E402
import pandas as pd # noqa: E402
import matplotlib# noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
 
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from scipy.spatial.distance import cdist, pdist, squareform  # noqa: E402
 
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
 
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN  # noqa: E402
from sklearn.manifold import MDS, TSNE  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
 
from sklearn.metrics import (  # noqa: E402
    silhouette_score, silhouette_samples,
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
)
from sklearn.metrics.cluster import contingency_matrix  # noqa: E402
 
from umap import UMAP  # noqa: E402
 
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
rng = np.random.default_rng(RANDOM_STATE)

import os  # noqa: E402
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

#Helper function

def sse(X, labels):
  """Sum of squared errors; ignores noise label -1."""
  total = 0.0
  for lbl in np.unique(labels):
    if lbl == -1:
      continue
      mask = labels == lbl
      total += np.sum((X[mask] - X[mask].mean(axis=0)) ** 2)

  return total

# --- Data Loading and Preprocessing ---

df_raw = pd.read_csv("healthcare-dataset-stroke-data.csv")


#Preprocessing
print("\n--- Preprocessing ---")
df = df_raw.copy()

df.drop(columns=["id"], inplace=True)

# Remove the "Other" Gender
df = df[df["gender"] != "Other"].reset_index(drop=True)

#Impute missing BMI with Median 
df["bmi"] = SimpleImputer(strategy="median").fit_transform(df[["bmi"]])

# Binary-encoding two-class categoricals
for col, mapping in [
  ("gender", {"Male": 1, "Female": 0}),
  ("ever_married", {"Yes": 1, "No": 0}),
  ("Residence_type", {"Urban": 1, "Rural": 0}) 
]:
  df[col] = df[col].map(mapping)

#One-hot encoding multi-class categoricals
df = pd.get_dummies(df, columns=["work_type", "smoking_status"], prefix=["work_type", "smoking_status"], drop_first=False)

stroke_labels = df["stroke"].values
df = df.drop(columns=["stroke"])

#Standardize numerical features
cont_cols = ["age", "avg_glucose_level", "bmi"]
df_scaled = df.copy().astype(float)
df_scaled[cont_cols] = StandardScaler().fit_transform(df_scaled[cont_cols])

X = df_scaled.values.astype(float)

# --- K - Means Clustering ---

print("\n--- K-Means Clustering ---")

sse_list ,sil_list, km_models = [], [], {}

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
axes[0].set_xlabel("k"); axes[0].set_ylabel("SSE")
axes[0].set_title("K-Means SSE Elbow Curve"); axes[0].legend()
 
axes[1].plot(range(2, 11), sil_list[1:], marker="s", color="darkorange")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
axes[1].set_title("K-Means Silhouette Scores")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "kmeans_elbow_silhouette.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: kmeans_elbow_silhouette.png")

# Best K = 4

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

# Per Sample silhouette plot
sil_vals = silhouette_samples(X, km_labels)
fig, ax  = plt.subplots(figsize=(10, 6))
cmap4    = plt.cm.get_cmap("tab10")
y_lower  = 10
 
for i in range(BEST_K):
  cluster_sil = np.sort(sil_vals[km_labels == i])
  y_upper = y_lower + cluster_sil.shape[0]
  ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, facecolor=cmap4(i / BEST_K), edgecolor=cmap4(i / BEST_K), alpha=0.7)
  ax.text(-0.05, y_lower + 0.5 * cluster_sil.shape[0], str(i), fontsize=10)
  y_lower = y_upper + 10
 
ax.axvline(silhouette_score(X, km_labels), color="red", linestyle="--", label="Average")
ax.set_xlabel("Silhouette Coefficient"); ax.set_ylabel("Cluster")
ax.set_title(f"Per-Sample Silhouette – K-Means (k={BEST_K})"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "silhouette_kmeans.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: silhouette_kmeans.png")


# --- Hierarchical Clustering ---

print("\n--- Hierarchical Clustering ---")

sample_idx = rng.choice(len(X), size=400, replace=False)
X_sub = X[sample_idx]

linkage_methods = ["ward", "complete", "average", "single"]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.ravel()

for i, method in enumerate(linkage_methods):
  Z = linkage(X_sub, method=method)
  dendrogram(Z, ax=axes[i], truncate_mode="lastp", p=25, leaf_rotation=90, leaf_font_size=7, color_threshold="default")
  axes[i].set_title(f"Dendrogram – {method.capitalize()} Linkage", fontsize=12)
  axes[i].set_xlabel("Cluster / data index"); axes[i].set_ylabel("Distance")

plt.suptitle("Hierarchical Clustering Dendrograms", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "hierarchical_dendrograms.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: hierarchical_dendrograms.png")

#Agglomerative k=4 on full data
hc_labels_dict = {}
print(f"\nAgglomerative Clustering (k={BEST_K}) on full dataset:")
print(f"  {'Method':<12} | {'SSE':>10} | {'Silhouette':>10} | Cluster sizes")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+------------------")

for method in linkage_methods:
  agg = AgglomerativeClustering(n_clusters=BEST_K, linkage=method)
  lbl = agg.fit_predict(X)
  hc_labels_dict[method] = lbl
  sizes = dict(zip(*np.unique(lbl, return_counts=True)))
  print(f"  {method:<12} | {sse(X,lbl):>10.2f} | {silhouette_score(X,lbl):>10.4f} | {sizes}")


# --- DBSCAN ---

print("\n--- DBSCAN Clustering ---")

# KNN Distance Plot
K_NN = 5
nbrs = NearestNeighbors(n_neighbors=K_NN).fit(X) 
distances, _ = nbrs.kneighbors(X)
k_dists = np.sort(distances[:, K_NN - 1])[::-1] 

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(k_dists, color="teal")
ax.axhline(y=1.5, color="red", linestyle="--", label="eps ≈ 1.5")
ax.set_xlabel("Data points (sorted)"); ax.set_ylabel(f"{K_NN}-NN Distance")
ax.set_title("k-NN Distance Plot – Estimating DBSCAN Epsilon"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "dbscan_knn_distance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: dbscan_knn_distance.png")

# Grid Search 
eps_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
min_pts_vals = [3,5,10,15]

best_cfg, best_sil = None, -np.inf

print("\n  eps  | min_pts | n_clusters | noise %  | Silhouette")
print("  -----+---------+------------+----------+-----------")

for eps in eps_values:
  for mpts in min_pts_vals:
    lbl   = DBSCAN(eps=eps, min_samples=mpts).fit_predict(X)
    n_cl  = len(set(lbl)) - (1 if -1 in lbl else 0)
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

print(f"  Clusters : {n_db}")
print(f"  Noise    : {noise_cnt} ({noise_cnt/len(X)*100:.1f}%)")
print(f"  Silhouette (non-noise): {silhouette_score(X, db_labels):.4f}")
print(f"  Cluster sizes: {dict(zip(*np.unique(db_labels, return_counts=True)))}")

# --- Anomaly Detection ---

print("\n--- Anomaly Detection ---")

THRES_HLD = 97

#K Means ---> Distance to Nearest Cluster
km_dist = np.min(cdist(X, km_centers), axis=1)
km_thresh = np.percentile(km_dist, THRES_HLD)
km_out_idx = np.where(km_dist > km_thresh)[0]
not_km_idx = np.where(km_dist <= km_thresh)[0]

print(f"\nK-Means Anomaly  (score = dist to nearest center)")
print(f"  {THRES_HLD}th-pct threshold : {km_thresh:.4f}")
print(f"  Outliers detected  : {len(km_out_idx)}")
print(f"  Stroke % outliers  : {stroke_labels[km_out_idx].mean()*100:.1f}%")
print(f"  Stroke % inliers   : {stroke_labels[not_km_idx].mean()*100:.1f}%")

df_viz["km_anomaly_score"] = km_dist
df_viz["is_km_outlier"] = (km_dist > km_thresh).astype(int)
print("\n  K-Means outlier vs. inlier continuous means:")
print(df_viz.groupby("is_km_outlier")[cont_cols].mean().round(2))

#Hierarchical Clustering(WARD)
ward_lbl   = hc_labels_dict["ward"]
ward_cent  = np.array([X[ward_lbl == i].mean(0) for i in range(BEST_K)])
ward_dist  = np.array([np.linalg.norm(X[i] - ward_cent[ward_lbl[i]]) for i in range(len(X))])
ward_thresh  = np.percentile(ward_dist, THRES_HLD)
ward_out_idx = np.where(ward_dist > ward_thresh)[0]
not_ward_idx = np.where(ward_dist <= ward_thresh)[0]

print(f"\nHierarchical (Ward) Anomaly  (score = dist to own centroid)")
print(f"  {THRES_HLD}th-pct threshold : {ward_thresh:.4f}")
print(f"  Outliers detected  : {len(ward_out_idx)}")
print(f"  Stroke % outliers  : {stroke_labels[ward_out_idx].mean()*100:.1f}%")
print(f"  Stroke % inliers   : {stroke_labels[not_ward_idx].mean()*100:.1f}%")

#DBSCAN Noise Label = -1
db_out_idx  = np.where(db_labels == -1)[0]
db_core_idx = np.where(db_labels != -1)[0]

print(f"\nDBSCAN Anomaly  (noise label = -1)")
print(f"  Noise points (outliers) : {len(db_out_idx)}")
if len(db_out_idx) > 0:
  print(f"  Stroke % outliers       : {stroke_labels[db_out_idx].mean()*100:.1f}%")
print(f"  Stroke % inliers        : {stroke_labels[db_core_idx].mean()*100:.1f}%")

#Overlap analysis
km_set   = set(km_out_idx)
ward_set = set(ward_out_idx)
db_set   = set(db_out_idx)
all3     = km_set & ward_set & db_set
km_ward  = km_set & ward_set

print(f"\nOutlier overlap:")
print(f"  K-Means ∩ Ward          : {len(km_ward)} points")
print(f"  All three methods       : {len(all3)} points")

if len(all3) > 0:
  all3_arr = np.array(list(all3))
  print(f"  Stroke rate (all-3)     : {stroke_labels[all3_arr].mean()*100:.1f}%")
  print(f"\n  All-3 outlier continuous feature stats:")
  print(df.iloc[all3_arr][cont_cols].describe().round(2))

#Anomaly Score Distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(km_dist, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
axes[0].axvline(km_thresh, color="red", linestyle="--", label=f"97th pct={km_thresh:.2f}")
axes[0].set_title("K-Means: Dist to Nearest Center")
axes[0].set_xlabel("Anomaly Score"); axes[0].legend()

axes[1].hist(ward_dist, bins=60, color="darkorange", edgecolor="white", alpha=0.8)
axes[1].axvline(ward_thresh, color="red", linestyle="--", label=f"97th pct={ward_thresh:.2f}")
axes[1].set_title("Hier (Ward): Dist to Own Centroid")
axes[1].set_xlabel("Anomaly Score"); axes[1].legend()

noise_mask = db_labels == -1
axes[2].bar(["Non-noise", "Noise (outlier)"],[(~noise_mask).sum(), noise_mask.sum()],color=["teal", "crimson"])
axes[2].set_title("DBSCAN: Noise vs Core/Border")
axes[2].set_ylabel("Count")

plt.suptitle("Anomaly Score Distributions", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "anomaly_score_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: anomaly_score_distributions.png")

# --- Internal Indices ---
print("\n--- Internal Indices ---")

print(f"\n  {'Method':<24} | {'SSE':>12} | {'Silhouette':>10}")
print(f"  {'-'*24}-+-{'-'*12}-+-{'-'*10}")

all_methods = [("K-Means k=4", km_labels)]
for m in linkage_methods:
  all_methods.append((f"Hier-{m} k=4", hc_labels_dict[m]))

nonnoise  = db_labels != -1
db_sil = (silhouette_score(X[nonnoise], db_labels[nonnoise]) if len(set(db_labels[nonnoise])) >= 2 else float("nan"))
all_methods.append(("DBSCAN", db_labels))

for name, lbl in all_methods:
  s_sse = sse(X, lbl)
  if name == "DBSCAN":
    s_sil = db_sil
  else:
    s_sil = silhouette_score(X, lbl)
  sil_str = f"{s_sil:.4f}" if not np.isnan(s_sil) else "  N/A "
  print(f"  {name:<24} | {s_sse:>12.2f} | {sil_str:>10}")

#Proximity Heatmap
HM_N    = 300
hm_idx  = rng.choice(len(X), HM_N, replace=False)
X_hm    = X[hm_idx]
lbl_hm  = km_labels[hm_idx]
order   = np.argsort(lbl_hm)
X_hm    = X_hm[order]; lbl_hm = lbl_hm[order]

dist_mat = squareform(pdist(X_hm, metric="euclidean"))
inc_mat  = (lbl_hm[:, None] == lbl_hm[None, :]).astype(float)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
im0 = axes[0].imshow(dist_mat, cmap="viridis_r", aspect="auto")
axes[0].set_title("Distance (Proximity) Matrix")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(inc_mat, cmap="Blues", aspect="auto")
axes[1].set_title("Incidence Matrix (same cluster = 1)")
plt.colorbar(im1, ax=axes[1])

plt.suptitle(f"K-Means k={BEST_K}: Proximity vs Incidence (n={HM_N})", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "proximity_incidence_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: proximity_incidence_heatmap.png")

# --- Relative Indices ---
print("\n--- Relative Indices ---")

def relative_metrics(la, lb, na, nb):
  ari = adjusted_rand_score(la, lb)
  nmi = normalized_mutual_info_score(la, lb)
  ami = adjusted_mutual_info_score(la, lb)
  print(f"\n  {na}  vs  {nb}")
  print(f"    ARI={ari:.4f}  NMI={nmi:.4f}  AMI={ami:.4f}")
  print(f"    SSE: {sse(X,la):.2f}  vs  {sse(X,lb):.2f}")

# 1. K-Means vs Hier-Ward (same k)
relative_metrics(km_labels, hc_labels_dict["ward"], "K-Means k=4", "Hier-Ward k=4")

# 2. K-Means two random seeds – tests stability
km_seed2 = KMeans(n_clusters=BEST_K, n_init=10, random_state=7).fit(X)
relative_metrics(km_labels, km_seed2.labels_, "K-Means seed=42", "K-Means seed=7")

# 3. Hier-Ward vs Hier-Complete
relative_metrics(hc_labels_dict["ward"], hc_labels_dict["complete"], "Hier-Ward k=4", "Hier-Complete k=4")

# 4. Hier-Average vs Hier-Single
relative_metrics(hc_labels_dict["average"], hc_labels_dict["single"], "Hier-Average k=4", "Hier-Single k=4")

# --- External Indices --- 
print("\n--- External Indices ---")

def external_metrics(lbl_pred, method_name):
  h, c, v = (homogeneity_score(stroke_labels, lbl_pred), completeness_score(stroke_labels, lbl_pred), v_measure_score(stroke_labels, lbl_pred))
  ari = adjusted_rand_score(stroke_labels, lbl_pred)
  nmi = normalized_mutual_info_score(stroke_labels, lbl_pred)
  ami = adjusted_mutual_info_score(stroke_labels, lbl_pred)
  print(f"\n  {method_name}")
  print(f"    Homogeneity={h:.4f}  Completeness={c:.4f}  V-Measure={v:.4f}")
  print(f"    ARI={ari:.4f}  NMI={nmi:.4f}  AMI={ami:.4f}")

external_metrics(km_labels, f"K-Means k={BEST_K}")
external_metrics(hc_labels_dict["ward"], f"Hierarchical Ward k={BEST_K}")
external_metrics(hc_labels_dict["complete"], f"Hierarchical Complete k={BEST_K}")
external_metrics(db_labels, "DBSCAN")

# Contigency Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (lbl, name) in zip(axes, [(km_labels, f"K-Means k={BEST_K}"), (hc_labels_dict["ward"], f"Hier-Ward k={BEST_K}"), (db_labels, "DBSCAN")]):
  cm = contingency_matrix(stroke_labels, lbl)
  sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax, xticklabels=[f"C{i}" for i in range(cm.shape[1])], yticklabels=["No Stroke", "Stroke"])
  ax.set_title(f"Contingency: {name}"); ax.set_xlabel("Cluster"); ax.set_ylabel("Stroke Label")

plt.suptitle("Contingency Matrices  (Clustering vs Stroke Label)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "contingency_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: contingency_matrices.png")

# --- Visulatization ---
print("\n--- Visualization ---")

# MDS, t-SNE, UMAP
VIS_N   = 500
vis_idx = rng.choice(len(X), VIS_N, replace=False)
X_vis   = X[vis_idx]

print("  Running MDS …")
X_mds  = MDS(n_components=2, random_state=RANDOM_STATE, dissimilarity="euclidean").fit_transform(X_vis)

print("  Running t-SNE …")
X_tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30).fit_transform(X_vis)

print("  Running UMAP …")
X_umap = UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=20, min_dist=0.1).fit_transform(X_vis)

print("  Plotting …")

lbl_vis = {
  "K-Means k=4"   : km_labels[vis_idx],
  "Hier-Ward k=4" : hc_labels_dict["ward"][vis_idx],
  "DBSCAN"        : db_labels[vis_idx],
  "Stroke Label"  : stroke_labels[vis_idx],
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
      mask  = labels == lv
      color = "grey" if lv == -1 else cmap_vis(j / max(len(uniq), 1))
      ax.scatter(X_proj[mask, 0], X_proj[mask, 1], c=[color], s=8, alpha=0.6, label=("Noise" if lv == -1 else str(lv)))
      ax.set_title(f"{proj_name} – {label_name}", fontsize=9)
      ax.set_xticks([]); ax.set_yticks([])
      if col == 0:
        ax.set_ylabel(proj_name, fontsize=11)
        if row == 0:
          ax.legend(markerscale=2, fontsize=7, loc="upper right")

plt.suptitle("Cluster Visualizations: MDS | t-SNE | UMAP  (n=500)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cluster_visualizations.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cluster_visualizations.png")