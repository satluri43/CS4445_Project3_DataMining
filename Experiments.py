import warnings
warnings.filterwarnings("ignore")

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from umap import UMAP

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =========================================================
# RESULTS LOG
# =========================================================
results = []

# LOAD DATA

df_raw = pd.read_csv("healthcare-dataset-stroke-data.csv")

def preprocess(df):
    df = df.copy()

    df = df.drop(columns=["id"])
    df = df[df["gender"] != "Other"]

    df["bmi"] = SimpleImputer(strategy="median").fit_transform(df[["bmi"]])

    df = pd.get_dummies(
        df,
        columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"],
        drop_first=False,
        dtype=int
    )

    df = df.drop(columns=["gender_Female", "ever_married_No", "Residence_type_Rural"])

    return df


def scale(df, cols):
    scaler = StandardScaler()
    df = df.copy()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def sse(X, labels):
    total = 0
    for l in np.unique(labels):
        if l == -1:
            continue
        mask = labels == l
        total += np.sum((X[mask] - X[mask].mean(axis=0)) ** 2)
    return total


def dbscan_analysis(labels):
    labels = np.array(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = np.sum(labels == -1)
    print(f"Clusters: {n_clusters}")
    print(f"Noise points: {noise} ({noise/len(labels)*100:.2f}%)")


def visualize_dbscan(X, labels, stroke=None, title="", filename=""):
    """
    2D projection + DBSCAN cluster visualization
    """

    # --- reduce to 2D (UMAP preferred, fallback PCA) ---
    try:
        X_2d = UMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    except:
        X_2d = PCA(n_components=2).fit_transform(X)

    labels = np.array(labels)

    plt.figure(figsize=(7, 5))

    unique_labels = np.unique(labels)

    cmap = plt.cm.get_cmap("tab10")

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl

        if lbl == -1:
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c="black",
                s=10,
                label="noise",
                alpha=0.6
            )
        else:
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                color=cmap(i % 10),
                s=10,
                label=f"C{lbl}",
                alpha=0.6
            )

    # --- optional stroke overlay ---
    if stroke is not None:
        stroke = np.array(stroke)
        plt.scatter(
            X_2d[stroke == 1, 0],
            X_2d[stroke == 1, 1],
            facecolors="none",
            edgecolors="red",
            s=40,
            label="stroke"
        )

    plt.title(title)
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved plot: {filename}")


def run_dbscan(name, X, eps, min_samples=5, stroke=None):
    print(f"\n=== {name} ===")
    t0 = time.time()
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = model.labels_
    t1 = time.time()

    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (labels == -1).sum()

    print(f"eps: {eps}")
    print(f"minPts: {min_samples}")
    print(f"# clusters: {n_clusters}")

    print("cluster distribution (%):")
    dist = {}
    for u, c in zip(unique, counts):
        label = "noise" if u == -1 else f"C{u}"
        pct = c / len(labels) * 100
        dist[label] = pct
        print(f"  {label}: {pct:.2f}%")

    runtime = t1 - t0
    print(f"time: {runtime:.3f}s")
    print(f"noise: {noise} ({noise/len(labels)*100:.2f}%)")

    non_noise = labels != -1
    sil = None
    if len(set(labels[non_noise])) > 1:
        sil = silhouette_score(X[non_noise], labels[non_noise])
        print(f"silhouette (non-noise): {sil:.4f}")

    stroke_noise = None
    stroke_core = None

    if stroke is not None:
        if noise > 0:
            stroke_noise = stroke[labels == -1].mean()
            print(f"stroke rate (noise): {stroke_noise:.4f}")

        stroke_core = stroke[labels != -1].mean()
        print(f"stroke rate (core): {stroke_core:.4f}")

    results.append({
        "experiment": name,
        "eps": eps,
        "minPts": min_samples,
        "clusters": n_clusters,
        "noise_pct": noise / len(labels) * 100,
        "runtime": runtime,
        "silhouette": sil,
        "stroke_noise": stroke_noise,
        "stroke_core": stroke_core,
        "distribution": dist
    })


    visualize_dbscan(
        X,
        labels,
        stroke=stroke,
        title=name,
        filename=name.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    )

    return labels

# =========================================================
# BASE DATA
# =========================================================
data = preprocess(df_raw)

num_cols = ["age", "avg_glucose_level", "bmi"]
data_scaled = scale(data, num_cols)

X = data_scaled.drop(columns=["stroke"]).values
y = data_scaled["stroke"].values



# =========================================================
# EXPERIMENT 1
# Extreme anomalies
# =========================================================
labels1 = run_dbscan("Experiment 1 (eps=1.75)", X, eps=1.75, stroke=y)


# =========================================================
# EXPERIMENT 2
# Stroke-only clustering
# =========================================================
stroke_only = data_scaled[data_scaled["stroke"] == 1].drop(columns=["stroke"])
X2 = stroke_only.values

labels2 = run_dbscan("Experiment 2 (stroke only)", X2, eps=1.3)


# =========================================================
# EXPERIMENT 3
# Age-based quantile clustering
# =========================================================
ex3 = data_scaled.copy()
stroke_ex3 = ex3["stroke"].values

ex3["age_group"] = pd.qcut(ex3["age"], q=3, labels=["young", "middle", "old"])

groups = {}

for group_name in ["young", "middle", "old"]:
    mask = ex3["age_group"] == group_name

    X_group = ex3[mask].drop(columns=["age_group", "stroke"]).values
    y_group = stroke_ex3[mask]

    groups[group_name] = (X_group, y_group)

for name, (Xg, yg) in groups.items():
    run_dbscan(
        f"Experiment 3 ({name})",
        Xg,
        eps=1.4,
        stroke=yg
    )


# =========================================================
# EXPERIMENT 4
# Feature reduction + smoking score
# =========================================================
ex4 = data_scaled.drop(columns=["work_type_Govt_job", "work_type_Never_worked", "work_type_Private"])

ex4["smoking_score"] = (
    0.0 * ex4["smoking_status_never smoked"] +
    0.5 * ex4["smoking_status_formerly smoked"] +
    0.5 * ex4["smoking_status_Unknown"] +
    1.0 * ex4["smoking_status_smokes"]
)

ex4 = ex4.drop(columns=[
    "smoking_status_never smoked",
    "smoking_status_formerly smoked",
    "smoking_status_smokes",
    "smoking_status_Unknown",
    "stroke"
])

X4 = ex4.values
labels4 = run_dbscan("Experiment 4 (feature reduction)", X4, eps=1.3, stroke=y)


# =========================================================
# EXPERIMENT 5
# Feature weighting
# =========================================================
ex5 = data_scaled.drop(columns=["stroke"]).copy()
ex5[["hypertension", "heart_disease"]] *= 3

X5 = ex5.values
labels5 = run_dbscan("Experiment 5 (weighted features)", X5, eps=1.6, stroke=y)

summary = pd.DataFrame(results)
print("\n================ FINAL EXPERIMENT SUMMARY ================\n")
print(summary[[
    "experiment",
    "eps",
    "minPts",
    "clusters",
    "noise_pct",
    "runtime",
    "silhouette",
    "stroke_core",
    "stroke_noise"
]])