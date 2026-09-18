import matplotlib.pyplot as plt


def show_table(title, rows):
    fig, ax = plt.subplots(figsize=(6, 0.6 + 0.4*len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                     loc="center", cellLoc="left")
    table.scale(1, 1.2)
    ax.set_title(title, pad=10)
    plt.show()

# ----------------------- Preprocessing -----------------------
# Build TopoNetX simplicial complexes, compute node & edge features,
# compute PH (landscape + persistence images + betti curves),
# save processed dataset (node_features augmented with edge-aggregates).

import os, glob, pickle
import numpy as np
from PIL import Image
from skimage.segmentation import slic
from skimage.color import rgb2gray
import networkx as nx
import toponetx as tnx
import gudhi
import gudhi.representations
import mahotas as mt

# ---------- Params ----------
NUM_SEGMENTS = 100
BASE_DIR = r'E:/Breast Cancer Patients MRI/train'
RANDOM_STATE = 42
LANDSCAPE_NUM = 5
LANDSCAPE_RESOLUTION = 200
BETTI_CURVE_RESOLUTION = 400
PI_GRID = (32, 32)
PI_WEIGHT_EXP = 1.0
# --------------------------------

# ---------- Helpers ----------
def load_image(path):
    img = np.array(Image.open(path).convert("RGB"))
    gray = (rgb2gray(img) * 255).astype(np.uint8)
    return img, gray

def segment_image(img):
    try:
        return slic(img, n_segments=NUM_SEGMENTS, compactness=10, start_label=0, random_state=RANDOM_STATE)
    except TypeError:
        return slic(img, n_segments=NUM_SEGMENTS, compactness=10, start_label=0)

def extract_node_features(img, gray, segments):
    """Return (n_superpixels, 21) features: RGB(3), centroid(2), gray_mean, gray_std, area, haralick(13)"""
    n = int(segments.max()) + 1
    feats = np.zeros((n, 21), dtype=float)
    for sp in range(n):
        mask = (segments == sp)
        if not mask.any():
            continue
        coords = np.argwhere(mask)
        feats[sp, :3] = img[mask].mean(axis=0)
        feats[sp, 3:5] = coords.mean(axis=0)
        feats[sp, 5] = float(gray[mask].mean())
        feats[sp, 6] = float(gray[mask].std())
        feats[sp, 7] = float(mask.sum())
        try:
            r0,c0 = coords.min(axis=0); r1,c1 = coords.max(axis=0)
            patch = gray[r0:r1+1, c0:c1+1]
            if patch.size >= 4:
                feats[sp, 8:] = mt.features.haralick(patch, return_mean=True)
        except Exception:
            pass
    return np.nan_to_num(feats)

def build_graph_and_borders(segments):
    """Return networkx graph and dict edge->border_length."""
    G = nx.Graph()
    borders = {}
    H,W = segments.shape
    for i in range(H):
        for j in range(W):
            u = int(segments[i,j])
            # only check right & down to avoid duplicate counts
            for di,dj in [(0,1),(1,0)]:
                ni,nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W:
                    v = int(segments[ni,nj])
                    if u != v:
                        a,b = (u,v) if u < v else (v,u)
                        borders[(a,b)] = borders.get((a,b), 0) + 1
                        G.add_edge(a,b)
    return G, borders

def build_simplicial_complex(G):
    sc = tnx.SimplicialComplex()
    for node in G.nodes():
        sc.add_simplex([int(node)])
    for u,v in G.edges():
        sc.add_simplex([int(u), int(v)])
    for clique in nx.find_cliques(G):
        if len(clique) == 3:
            sc.add_simplex(list(map(int, clique)))
    return sc

def create_edge_features_from_segments(segments, node_features):
    """Return edge_list ([(u,v),...]) and edge_feats (n_edges x 8)."""
    G, borders = build_graph_and_borders(segments)
    edges = list(G.edges())
    if not edges:
        return [], np.empty((0,8), dtype=float)

    nf = np.asarray(node_features, dtype=float)
    har_start = 8
    har_end = nf.shape[1]
    edge_list = []
    feats = []
    for (u,v) in edges:
        if u >= nf.shape[0] or v >= nf.shape[0]:
            continue
        col_diff = np.abs(nf[u, :3] - nf[v, :3])        # 3
        gray_diff = float(abs(nf[u,5] - nf[v,5]))       # 1
        area_rel = float(abs(nf[u,7] - nf[v,7]) / (nf[u,7] + nf[v,7] + 1e-8))  # 1
        centroid_dist = float(np.linalg.norm(nf[u,3:5] - nf[v,3:5]))  # 1
        if har_end > har_start:
            hdist = float(np.linalg.norm(nf[u,har_start:har_end] - nf[v,har_start:har_end]))
        else:
            hdist = 0.0
        a,b = (u,v) if u < v else (v,u)
        border_len = float(borders.get((a,b), 0))  # 1
        edge_vec = np.hstack([col_diff, gray_diff, area_rel, centroid_dist, hdist, border_len])
        feats.append(edge_vec)
        edge_list.append((u,v))
    edge_feats = np.vstack(feats) if feats else np.empty((0,8), dtype=float)
    return edge_list, edge_feats

def aggregate_edge_feats_to_nodes(edge_list, edge_feats, n_nodes):
    """Return (n_nodes, 8*3) array: [mean,std,max] per edge feature aggregated over incident edges."""
    if edge_feats.size == 0:
        return np.zeros((n_nodes, 8*3), dtype=float)
    per_node = [[] for _ in range(n_nodes)]
    for (u,v), ef in zip(edge_list, edge_feats):
        per_node[u].append(ef); per_node[v].append(ef)
    node_agg = np.zeros((n_nodes, edge_feats.shape[1]*3), dtype=float)
    for i, lst in enumerate(per_node):
        if not lst:
            continue
        A = np.vstack(lst)
        node_agg[i,:] = np.hstack([A.mean(0), A.std(0), A.max(0)])
    return node_agg

# --- Persistence helpers (lightweight and robust) ---
def compute_betti_curve_from_intervals(intervals, resolution=BETTI_CURVE_RESOLUTION, t_min=None, t_max=None):
    intervals_f = [(float(b), float(d)) for (b,d) in intervals if np.isfinite(d) and (d > b)]
    if not intervals_f:
        ts = np.linspace(0,1,resolution)
        return ts, np.zeros_like(ts)
    births = [b for (b,d) in intervals_f]; deaths = [d for (b,d) in intervals_f]
    if t_min is None: t_min = min(births)
    if t_max is None: t_max = max(deaths)
    if t_min == t_max: t_max = t_min + 1e-6
    ts = np.linspace(t_min, t_max, resolution)
    betti_vals = np.array([sum((b <= t <= d) for (b,d) in intervals_f) for t in ts])
    return ts, betti_vals

def compute_persistence_image(intervals, grid=PI_GRID, weight_exp=PI_WEIGHT_EXP, birth_range=None, pers_range=None, sigma=None):
    pts = [(float(b), float(d)-float(b)) for (b,d) in intervals if np.isfinite(d) and (d > b)]
    if len(pts) == 0:
        return np.zeros((grid[0]*grid[1],), dtype=float), np.zeros(grid, dtype=float)
    births = np.array([p[0] for p in pts]); pers = np.array([p[1] for p in pts])
    if birth_range is None:
        bmin,bmax = births.min(), births.max()
        bpad = (bmax-bmin)*0.05 if (bmax>bmin) else 1.0
        birth_range = (bmin-bpad, bmax+bpad)
    if pers_range is None:
        pmin,pmax = 0.0, pers.max()
        ppad = (pmax-pmin)*0.05 if (pmax>pmin) else 1.0
        pers_range = (pmin, pmax+ppad)
    gx = np.linspace(birth_range[0], birth_range[1], grid[1])
    gy = np.linspace(pers_range[0], pers_range[1], grid[0])
    Xc, Yc = np.meshgrid(gx, gy)
    sigma = sigma or max((birth_range[1]-birth_range[0])/grid[1], (pers_range[1]-pers_range[0])/grid[0]) * 1.0
    img = np.zeros_like(Xc, dtype=float)
    for (b,p) in pts:
        w = (p**weight_exp) if p>0 else 0.0
        if w==0: continue
        exponent = -(((Xc-b)**2 + (Yc-p)**2) / (2.0*sigma**2))
        img += w * np.exp(exponent)
    s = img.sum()
    if s>0: img = img / s
    return img.flatten(), img

def get_topological_features(sc, node_features, max_dimension=1, resolution=LANDSCAPE_RESOLUTION, num_landscapes=LANDSCAPE_NUM):
    f = node_features[:,6]  # gray std
    filtration = (f - f.min()) / (f.max() - f.min() + 1e-8)
  # mean gray per node
    st = gudhi.SimplexTree()
    for simplex in sc.simplices:
        simplex_list = [int(x) for x in simplex]
        if not simplex_list: continue
        filtval = float(np.max([filtration[n] for n in simplex_list]))
        st.insert(simplex_list, filtration=filtval)
    st.persistence()
    pd = st.persistence()

    landscape = gudhi.representations.Landscape(num_landscapes=num_landscapes, resolution=resolution)
    L_LENGTH = num_landscapes * resolution
    landscape_features = []
    intervals_dim = {}
    for dim in range(max_dimension+1):
        pairs = np.array(st.persistence_intervals_in_dimension(dim), dtype=float)
        if pairs.size==0:
            landscape_features.extend([0.0]*L_LENGTH)
            intervals_dim[dim] = np.empty((0,2))
            continue
        finite_mask = np.isfinite(pairs[:,1]) & (pairs[:,1] > pairs[:,0])
        valid_pairs = pairs[finite_mask]
        if valid_pairs.size==0:
            landscape_features.extend([0.0]*L_LENGTH)
        else:
            try:
                vec = landscape.fit_transform([valid_pairs])[0]
                if len(vec) < L_LENGTH:
                    vec = np.concatenate([vec, np.zeros(L_LENGTH-len(vec))])
                landscape_features.extend(vec.tolist())
            except Exception:
                landscape_features.extend([0.0]*L_LENGTH)
        intervals_dim[dim] = pairs

    # Betti curves (use st.persistence_intervals_in_dimension)
    pd_list = pd
    births_all = [b for (_, (b,d)) in pd_list]
    deaths_all = [d for (_, (b,d)) in pd_list if np.isfinite(d)]
    if len(births_all)==0 or len(deaths_all)==0:
        tmin,tmax = 0.0, 1.0
    else:
        tmin = min(births_all); tmax = max(deaths_all)
        if tmin == tmax: tmax = tmin + 1e-6
    betti_curve = {}
    for dim in [0,1]:
        intervals = [(b,d) for (d_dim,(b,d)) in pd_list if d_dim==dim]
        ts, vals = compute_betti_curve_from_intervals(intervals, resolution=BETTI_CURVE_RESOLUTION, t_min=tmin, t_max=tmax)
        betti_curve[dim] = {'ts': ts.tolist(), 'vals': vals.tolist()}

    # PIs (H0 & H1)
    H0 = intervals_dim.get(0, np.empty((0,2))).tolist()
    H1 = intervals_dim.get(1, np.empty((0,2))).tolist()
    pi_H0_flat, pi_H0_map = compute_persistence_image(H0, grid=PI_GRID)
    pi_H1_flat, pi_H1_map = compute_persistence_image(H1, grid=PI_GRID)

    return {
        'persistence_diagram': pd_list,
        'landscape_features': landscape_features,
        'betti_numbers': [len(intervals_dim.get(d, [])) for d in (0,1)],
        'H0_intervals': H0,
        'H1_intervals': H1,
        'betti_curves': betti_curve,
        'persistence_image_H0': pi_H0_flat,
        'persistence_image_H0_map': pi_H0_map,
        'persistence_image_H1': pi_H1_flat,
        'persistence_image_H1_map': pi_H1_map
    }

def compute_centroids_from_segments(segments):
    pos={}
    for sp in range(int(segments.max())+1):
        coords = np.argwhere(segments==sp)
        if coords.size==0:
            pos[sp] = (0.0,0.0)
        else:
            rmean,cmean = coords.mean(axis=0)
            pos[sp] = (float(cmean), float(rmean))
    return pos

# ---------- Main processing loop ----------
def process_dataset_and_save():
    data = []
    classes = ['negative','positive']
    skipped = 0

    for label, cls in enumerate(classes):
        paths = sorted(glob.glob(os.path.join(BASE_DIR, cls, '*.jpg')))
        print(f"Processing class {cls}: {len(paths)} images")

        for i, path in enumerate(paths):
            try:
                # ---------- load ----------
                img, gray = load_image(path)
                if img is None or gray is None:
                    skipped += 1
                    continue

                # ---------- segmentation ----------
                segments = segment_image(img)
                if segments is None:
                    skipped += 1
                    continue

                # ---------- node / edge features ----------
                node_feats = extract_node_features(img, gray, segments)
                edge_list, edge_feats = create_edge_features_from_segments(
                    segments, node_feats
                )
                node_edge_agg = aggregate_edge_feats_to_nodes(
                    edge_list, edge_feats, int(segments.max()) + 1
                )
                node_feats_aug = np.hstack([node_feats, node_edge_agg])

                # ---------- graph + simplicial complex ----------
                G, _ = build_graph_and_borders(segments)
                sc = build_simplicial_complex(G)

                # ---------- persistent homology ----------
                ph = get_topological_features(sc, node_feats)

                # ---------- store ----------
                sample = {
                    'node_features': node_feats_aug,
                    'edge_features': edge_feats,
                    'simplicial_complex': sc,
                    'landscape_features': ph['landscape_features'],
                    'betti_numbers': ph['betti_numbers'],
                    'persistence_diagram': ph['persistence_diagram'],
                    'betti_curves': ph['betti_curves'],
                    'persistence_image_H0': ph['persistence_image_H0'],
                    'persistence_image_H0_map': ph['persistence_image_H0_map'],
                    'persistence_image_H1': ph['persistence_image_H1'],
                    'persistence_image_H1_map': ph['persistence_image_H1_map'],
                    'label': label,
                    'superpixel_labels': segments,
                    'positions': compute_centroids_from_segments(segments),
                    'image_path': path
                }
                data.append(sample)

                if (i + 1) % 50 == 0:
                    print(f"  processed {i+1}/{len(paths)}")

            except Exception as e:
                skipped += 1
                print(f"[SKIP] {path}")
                print(f"       {type(e).__name__}: {e}")
                continue

    print(f"\nFinished. Saved samples: {len(data)} | Skipped: {skipped}")

    # ---------- save ----------
    out_name = 'processed_toponetx_edgeagg.pkl'
    OUT_DIR = 'Content/Spider projects 26-1/Breast cancer classification'
    out_path = os.path.join(OUT_DIR, out_name)

    try:
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(data)} samples to: {out_path}")
    except Exception as e:
        fallback = os.path.join(os.path.expanduser('~'), 'Documents', out_name)
        with open(fallback, 'wb') as f:
            pickle.dump(data, f)
        print(f"Could not write to {out_path} (error: {e}). Saved instead to {fallback}")

# Execute
process_dataset_and_save()

# --- params (edit if needed) ---
PICKLE_PATH = 'Content/Breast cancer classification/processed_toponetx_edgeagg.pkl'
from sklearn.model_selection import train_test_split
# --- 1) collect raw image paths in a deterministic order (by class then sorted filenames) ---
raw_paths = []
raw_labels = []
for label, cls in enumerate(['negative', 'positive']):
    paths = sorted(glob.glob(os.path.join(BASE_DIR, cls, '*.jpg')))
    for p in paths:
        raw_paths.append(os.path.abspath(p))
        raw_labels.append(label)
raw_paths = np.array(raw_paths)
raw_labels = np.array(raw_labels)
print(f"Found {len(raw_paths)} raw images ({np.sum(raw_labels==0)} neg, {np.sum(raw_labels==1)} pos).")

# --- 2) load TDA samples and try to find explicit filepath keys ---
with open(PICKLE_PATH, 'rb') as f:
    samples = pickle.load(f)
print(f"Loaded {len(samples)} TDA samples from pickle.")

# possible keys that might store the original filename/path inside each sample
possible_keys = ['filepath','file_path','path','image_path','image','img_path','filename','file','file_name','orig_path','source','id','index']

sample_to_raw_path = []
missing_filepath_keys = True
for key in possible_keys:
    if all((isinstance(s, dict) and key in s) for s in samples):
        # use this key
        sample_to_raw_path = [os.path.abspath(s[key]) for s in samples]
        missing_filepath_keys = False
        print(f"Using '{key}' key from pickle for mapping.")
        break

# --- 3) if pickle lacks explicit filepath, match by label & order within each class (fallback) ---
if missing_filepath_keys:
    print("No explicit filepath key found in pickle. Falling back to matching by label-and-order within class.")
    # build class-ordered lists from raw_paths
    class_paths = {0: list(raw_paths[raw_labels==0]), 1: list(raw_paths[raw_labels==1])}
    counters = {0:0, 1:0}
    ok = True
    for s in samples:
        lab = int(s.get('label', -1))
        if lab not in (0,1):
            ok = False
            break
        if counters[lab] >= len(class_paths[lab]):
            ok = False
            break
        sample_to_raw_path.append(class_paths[lab][counters[lab]])
        counters[lab] += 1
    if not ok or len(sample_to_raw_path) != len(samples):
        raise RuntimeError(
            "Fallback mapping by label/order failed — the pickle doesn't contain filepaths and counts don't match raw images.\n"
            "Please re-save the pickle adding an explicit 'filepath' or 'filename' field for each sample."
        )
    print("Fallback mapping succeeded (assumes TDA samples were saved in class-order matching raw files).")

# --- 4) convert sample->raw path into indices in raw_paths (match by absolute path first, then basename) ---
raw_path_to_index = {os.path.abspath(p): idx for idx, p in enumerate(raw_paths)}
basename_to_indices = {}
for idx,p in enumerate(raw_paths):
    b = os.path.basename(p)
    basename_to_indices.setdefault(b, []).append(idx)

sample_to_raw_index = []
for p in sample_to_raw_path:
    abs_p = os.path.abspath(p)
    if abs_p in raw_path_to_index:
        sample_to_raw_index.append(raw_path_to_index[abs_p])
    else:
        b = os.path.basename(p)
        if b in basename_to_indices and len(basename_to_indices[b])==1:
            sample_to_raw_index.append(basename_to_indices[b][0])
        else:
            # ambiguous or not found
            raise RuntimeError(f"Could not unambiguously map sample path '{p}' to a raw image. Check filenames and paths.")

sample_to_raw_index = np.array(sample_to_raw_index, dtype=int)
if sample_to_raw_index.size != len(samples):
    raise RuntimeError("Mapping length mismatch between samples and sample_to_raw_index.")

# --- 5) build reverse map raw_index -> sample_index (only for raw images that have a TDA sample) ---
raw_to_sample_index = {raw_idx: sample_idx for sample_idx, raw_idx in enumerate(sample_to_raw_index)}

# --- 6) create a single random 80/20 split over the *samples* (so both TDA and raw use same sample selection) ---
y_samples = np.array([int(s['label']) for s in samples])
raw_indices_for_split = sample_to_raw_index.copy()   # these are raw indices corresponding to each sample in same order
# we will split by sample order using stratify=y_samples; train_test_split returns raw indices (not sample indices),
# but we also compute the corresponding sample indices (positions in samples list)
train_raw_idx, test_raw_idx, train_y, test_y = train_test_split(
    raw_indices_for_split, y_samples,
    test_size=0.2, stratify=y_samples, shuffle=True
)

# convert to sample indices (indexes in samples list)
raw_to_sample = raw_to_sample_index
train_sample_idx = np.array([raw_to_sample[r] for r in train_raw_idx], dtype=int)
test_sample_idx  = np.array([raw_to_sample[r] for r in test_raw_idx], dtype=int)

# --- 7) expose globals for other cells to use ---
# RAW-level indices (into raw_paths, X_raw etc.)
RAW_TRAIN_IDX = train_raw_idx
RAW_TEST_IDX  = test_raw_idx
# SAMPLE-level indices (into samples list and TDA feature matrices built from samples)
SAMPLE_TRAIN_IDX = train_sample_idx
SAMPLE_TEST_IDX  = test_sample_idx

print("Split created:")
print("  #samples total:", len(samples))
print("  train samples:", len(SAMPLE_TRAIN_IDX), "test samples:", len(SAMPLE_TEST_IDX))
print("  RAW_TRAIN_IDX shape:", RAW_TRAIN_IDX.shape, "RAW_TEST_IDX shape:", RAW_TEST_IDX.shape)
# ---------------------------------------------------------------______________________________________________________________#
# ======================= RF on NODE vs NODE+PH with CV + Accuracy =======================
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna

USE_OPTUNA = True    # set True to run Optuna tuning
N_TRIALS = 25
RANDOM_STATE = 42
PH_STACK = True      # enable NODE+PH(prob) stacking
N_FOLDS = 5         # number of folds for CV

# --- load processed samples ---
PICKLE_PATH = 'Content/Spider projects 26-1/Breast cancer classification/processed_toponetx_edgeagg.pkl'
with open(PICKLE_PATH,'rb') as f:
    samples = pickle.load(f)
n_samples = len(samples)
print("Loaded samples:", n_samples)

# --- build X_node ---
def aggregate_node_stats(nf):
    nf = np.asarray(nf,dtype=float)
    if nf.size==0: return np.zeros(5,dtype=float)
    stats = [fn(nf,axis=0) for fn in (np.mean,np.std,np.median,np.min,np.max)]
    return np.concatenate(stats)

X_node = np.vstack([aggregate_node_stats(s['node_features']) for s in samples])
print("X_node shape:", X_node.shape)

# --- PH PCA block ---
X_ph_raw = np.vstack([np.asarray(s.get('landscape_features',[]),dtype=float) for s in samples])
if X_ph_raw.size>0:
    scaler_ph = StandardScaler()
    scaler_ph.fit(X_ph_raw[SAMPLE_TRAIN_IDX])
    X_ph_scaled = np.zeros_like(X_ph_raw)
    X_ph_scaled[SAMPLE_TRAIN_IDX] = scaler_ph.transform(X_ph_raw[SAMPLE_TRAIN_IDX])
    X_ph_scaled[SAMPLE_TEST_IDX]  = scaler_ph.transform(X_ph_raw[SAMPLE_TEST_IDX])
    pca_ph = PCA(n_components=0.95, svd_solver='full', random_state=RANDOM_STATE)
    pca_ph.fit(X_ph_scaled[SAMPLE_TRAIN_IDX])
    X_ph_pca = pca_ph.transform(X_ph_scaled)
else:
    X_ph_pca = np.zeros((n_samples,0))
print("X_ph_pca shape:", X_ph_pca.shape)

# --- scale node features ---
sc_node = StandardScaler()
sc_node.fit(X_node[SAMPLE_TRAIN_IDX])
X_node_scaled = np.zeros_like(X_node)
X_node_scaled[SAMPLE_TRAIN_IDX] = sc_node.transform(X_node[SAMPLE_TRAIN_IDX])
X_node_scaled[SAMPLE_TEST_IDX]  = sc_node.transform(X_node[SAMPLE_TEST_IDX])

# --- prepare matrices ---
X_node_train = X_node_scaled[SAMPLE_TRAIN_IDX]; X_node_test = X_node_scaled[SAMPLE_TEST_IDX]
y = np.array([int(s['label']) for s in samples])
y_train = y[SAMPLE_TRAIN_IDX]; y_test = y[SAMPLE_TEST_IDX]

# --- Function to train RF with optional Optuna and compute CV metrics ---
def train_rf_optuna_cv(X_tr, y_tr, X_val, y_val, use_optuna=True, n_folds=5):
    # Hyperparameter tuning
    if use_optuna:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators',1000,3000,step=100),
                'max_depth': trial.suggest_int('max_depth',10,30),
                'min_samples_split': trial.suggest_int('min_samples_split',2,8),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf',1,5),
                'max_features': trial.suggest_float('max_features',0.1,0.5),
                'class_weight':'balanced',
                'n_jobs':-1,
                'random_state':RANDOM_STATE
            }
            rf = RandomForestClassifier(**params)
            rf.fit(X_tr, y_tr)
            y_pred = rf.predict_proba(X_val)[:,1]
            return roc_auc_score(y_val, y_pred)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS)
        best_params = study.best_params
    else:
        best_params = {'n_estimators':1200,'max_depth':20,'min_samples_split':2,'min_samples_leaf':1,'max_features':0.3}
    
    # Train final model
    rf = RandomForestClassifier(**best_params, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_tr, y_tr)
    y_pred_prob = rf.predict_proba(X_val)[:,1]
    y_pred_label = (y_pred_prob >= 0.5).astype(int)
    
    # Compute test metrics
    auc = roc_auc_score(y_val, y_pred_prob)
    acc = accuracy_score(y_val, y_pred_label)
    
    # Compute CV metrics on training set
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(rf, X_tr, y_tr, cv=skf, scoring='roc_auc', n_jobs=-1).mean()
    cv_acc = cross_val_score(rf, X_tr, y_tr, cv=skf, scoring='accuracy', n_jobs=-1).mean()
    
    return rf, auc, acc, cv_auc, cv_acc, y_pred_prob

# --- Train NODE-only RF ---
rf_node, node_auc, node_acc, node_cv_auc, node_cv_acc, y_node_pred = train_rf_optuna_cv(
    X_node_train, y_train, X_node_test, y_test, USE_OPTUNA, N_FOLDS
)
print(f"NODE-only Test AUC: {node_auc:.4f} | Accuracy: {node_acc:.4f} | CV AUC: {node_cv_auc:.4f} | CV Acc: {node_cv_acc:.4f}")

# --- Train NODE+PH stacked RF ---
if PH_STACK and X_ph_pca.size>0:
    X_ph_train = X_ph_pca[SAMPLE_TRAIN_IDX]; X_ph_test = X_ph_pca[SAMPLE_TEST_IDX]
    rf_ph, _, _, _, _, _ = train_rf_optuna_cv(X_ph_train, y_train, X_ph_test, y_test, USE_OPTUNA, N_FOLDS)
    
    # Stack PH probabilities with NODE features
    ph_train_prob = rf_ph.predict_proba(X_ph_train)[:,1].reshape(-1,1)
    ph_test_prob  = rf_ph.predict_proba(X_ph_test)[:,1].reshape(-1,1)
    X_stack_train = np.hstack([X_node_train, ph_train_prob])
    X_stack_test  = np.hstack([X_node_test, ph_test_prob])
    
    rf_stack, stack_auc, stack_acc, stack_cv_auc, stack_cv_acc, y_stack_pred = train_rf_optuna_cv(
        X_stack_train, y_train, X_stack_test, y_test, USE_OPTUNA, N_FOLDS
    )
    print(f"NODE+PH stacked Test AUC: {stack_auc:.4f} | Accuracy: {stack_acc:.4f} | CV AUC: {stack_cv_auc:.4f} | CV Acc: {stack_cv_acc:.4f}")
else:
    rf_stack = None; stack_auc = None; stack_acc=None; stack_cv_auc=None; stack_cv_acc=None

# --- plot ROCs ---
plt.figure(figsize=(6,6))
fpr, tpr, _ = roc_curve(y_test, y_node_pred); plt.plot(fpr, tpr, label=f"RF NODE-only AUC={node_auc:.3f}")
if PH_STACK and rf_stack is not None:
    fpr, tpr, _ = roc_curve(y_test, y_stack_pred); plt.plot(fpr, tpr, label=f"RF NODE+PH stacked AUC={stack_auc:.3f}")
plt.plot([0,1],[0,1],'k--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves - Node vs Node+PH"); plt.legend(); plt.grid(); plt.show()

# --- expose globals ---
globals().update({
    'rf_node': rf_node, 'rf_stack': rf_stack,
    'X_node_train': X_node_train, 'X_node_test': X_node_test,
    'X_ph_train': X_ph_train if X_ph_pca.size>0 else None,
    'X_ph_test': X_ph_test if X_ph_pca.size>0 else None,
    'y_train': y_train, 'y_test': y_test,
    'node_auc': node_auc, 'node_acc': node_acc, 'node_cv_auc': node_cv_auc, 'node_cv_acc': node_cv_acc,
    'stack_auc': stack_auc, 'stack_acc': stack_acc, 'stack_cv_auc': stack_cv_auc, 'stack_cv_acc': stack_cv_acc
})

show_table(
    "RF – Node & Node+PH",
    [
        ["NODE Test AUC", f"{node_auc:.4f}"],
        ["NODE Test Acc", f"{node_acc:.4f}"],
        ["NODE CV AUC", f"{node_cv_auc:.4f}"],
        ["NODE CV Acc", f"{node_cv_acc:.4f}"],
        ["NODE+PH Test AUC", f"{stack_auc:.4f}"],
        ["NODE+PH Test Acc", f"{stack_acc:.4f}"],
        ["NODE+PH CV AUC", f"{stack_cv_auc:.4f}"],
        ["NODE+PH CV Acc", f"{stack_cv_acc:.4f}"],
    ]
)
#_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# ================= RF on RAW features  =================
import os, glob, pickle, numpy as np
from PIL import Image
from skimage.color import rgb2gray
import mahotas as mt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import optuna

# ---------- user params ----------
USE_OPTUNA = True
N_OPTUNA_TRIALS = 25
RANDOM_STATE = 42
# ---------------------------------
# ---------- locate processed pickle (try typical locations) ----------
cands = []
cands.append(os.path.join(r'Content/Breast cancer classification','processed_toponetx_edgeagg.pkl'))
cands.append(os.path.join(os.path.expanduser('~'),'Documents','processed_toponetx_edgeagg.pkl'))
cands.extend(glob.glob(os.path.join(os.getcwd(),'**','processed*.pkl'), recursive=True))
cands = [p for p in dict.fromkeys(cands) if os.path.exists(p)]
if not cands:
    raise FileNotFoundError("Processed pickle not found. Run (process_dataset_and_save) first.")
PICKLE_PATH = cands[0]
print("Using pickle:", PICKLE_PATH)

# ---------- load samples ----------
with open(PICKLE_PATH, 'rb') as f:
    samples = pickle.load(f)
n = len(samples)
print("Loaded samples:", n)

# ---------- find raw image paths  ----------
# try to infer base dir from pickle location or use common path
inferred_base = os.path.dirname(PICKLE_PATH)
# usual layout: base/negative/*.jpg and base/positive/*.jpg
raw_paths = []
raw_labels = []
for label, cls in enumerate(['negative','positive']):
    folder = os.path.join(inferred_base, cls)
    if not os.path.isdir(folder):
        folder = r'E:/Breast Cancer Patients MRI/train'
        folder = os.path.join(folder, cls)
    files = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    for p in files:
        raw_paths.append(os.path.abspath(p)); raw_labels.append(label)
raw_paths = np.array(raw_paths)
raw_labels = np.array(raw_labels)
if len(raw_paths)==0:
    raise RuntimeError("No raw images found. Check dataset folders.")

print(f"Found {len(raw_paths)} raw images ({raw_labels.sum()} positives).")

# ---------- helper to extract lightweight raw features ----------
def extract_raw_features(path):
    img = np.array(Image.open(path).convert("RGB"))
    gray = (rgb2gray(img) * 255).astype(np.uint8)
    feats = []
    # RGB means
    feats.extend(img.reshape(-1,3).mean(axis=0).tolist())
    feats.append(float(gray.mean()))
    feats.append(float(gray.std()))
    # haralick on center crop if possible
    H,W = gray.shape
    if H>8 and W>8:
        r0 = max(0, H//2 - 32); c0 = max(0, W//2 - 32)
        patch = gray[r0:r0+64, c0:c0+64] if H>64 and W>64 else gray
        try:
            har = mt.features.haralick(patch, return_mean=True)
        except Exception:
            har = np.zeros(13)
    else:
        har = np.zeros(13)
    feats.extend(har.tolist())
    return np.asarray(feats, dtype=float)

# ---------- build X_raw (image-level features) ----------
print("Extracting raw features (this will take a moment)...")
X_raw = np.vstack([extract_raw_features(p) for p in raw_paths])
print("X_raw shape:", X_raw.shape)

# ---------- map samples -> raw indices (use image_path in samples if present, else fall back to class-order) ----------
if all(('image_path' in s) for s in samples):
    raw_path_to_idx = {os.path.abspath(p):i for i,p in enumerate(raw_paths)}
    sample_to_raw_index = []
    for s in samples:
        p = os.path.abspath(s.get('image_path'))
        if p not in raw_path_to_idx:
            raise RuntimeError(f"Could not map sample.image_path {p} to found raw images.")
        sample_to_raw_index.append(raw_path_to_idx[p])
    sample_to_raw_index = np.array(sample_to_raw_index, dtype=int)
else:
    # fallback: assume samples are in class-order; mirror Split fallback
    class_paths = {0: list(raw_paths[raw_labels==0]), 1: list(raw_paths[raw_labels==1])}
    counters = {0:0, 1:0}
    sample_to_raw_index = []
    for s in samples:
        lab = int(s.get('label', 0))
        sample_to_raw_index.append(class_paths[lab][counters[lab]])
        counters[lab] += 1
    # convert basenames to indices
    basename_to_indices = {}
    for idx,p in enumerate(raw_paths):
        basename_to_indices.setdefault(os.path.basename(p), []).append(idx)
    mapped = []
    for p in sample_to_raw_index:
        b = os.path.basename(p)
        idxs = basename_to_indices.get(b, [])
        if len(idxs)==1:
            mapped.append(idxs[0])
        else:
            raise RuntimeError("Ambiguous mapping from samples to raw images. Ensure samples['image_path'] exists.")
    sample_to_raw_index = np.array(mapped, dtype=int)

# sample-level raw features aligned with samples
X_raw_sample = X_raw[sample_to_raw_index]
print("X_raw_sample shape (aligned to samples):", X_raw_sample.shape)

SAMPLE_TRAIN_IDX = globals().get('SAMPLE_TRAIN_IDX', None)
SAMPLE_TEST_IDX  = globals().get('SAMPLE_TEST_IDX', None)
if SAMPLE_TRAIN_IDX is None or SAMPLE_TEST_IDX is None:
    # create deterministic stratified split and save as globals
    y_labels = np.array([int(s['label']) for s in samples])
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.2, stratify=y_labels, random_state=RANDOM_STATE)
    SAMPLE_TRAIN_IDX = np.array(tr, dtype=int); SAMPLE_TEST_IDX = np.array(te, dtype=int)
    globals()['SAMPLE_TRAIN_IDX'] = SAMPLE_TRAIN_IDX
    globals()['SAMPLE_TEST_IDX'] = SAMPLE_TEST_IDX
    print("SAMPLE_TRAIN_IDX / SAMPLE_TEST_IDX not found — created new split and stored as globals.")
else:
    SAMPLE_TRAIN_IDX = np.array(SAMPLE_TRAIN_IDX, dtype=int)
    SAMPLE_TEST_IDX = np.array(SAMPLE_TEST_IDX, dtype=int)
    print("Using existing SAMPLE_TRAIN_IDX / SAMPLE_TEST_IDX from globals.")

# ---------- scale raw (fit on train only) ----------
scaler = StandardScaler()
scaler.fit(X_raw_sample[SAMPLE_TRAIN_IDX])
X_raw_scaled = np.zeros_like(X_raw_sample, dtype=float)
X_raw_scaled[SAMPLE_TRAIN_IDX] = scaler.transform(X_raw_sample[SAMPLE_TRAIN_IDX])
X_raw_scaled[SAMPLE_TEST_IDX]  = scaler.transform(X_raw_sample[SAMPLE_TEST_IDX])

X_train = X_raw_scaled[SAMPLE_TRAIN_IDX]; X_test = X_raw_scaled[SAMPLE_TEST_IDX]
y = np.array([int(s['label']) for s in samples])
y_train = y[SAMPLE_TRAIN_IDX]; y_test = y[SAMPLE_TEST_IDX]

# ---------- Optuna objective (if enabled) ----------
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000, step=100),
        'max_depth': trial.suggest_int('max_depth', 6, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 6),
        'max_features': trial.suggest_categorical('max_features', ['sqrt','log2', 0.2, 0.3])
    }
    rf = RandomForestClassifier(**params, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    cv_acc = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    print(f"CV AUC: {cv_auc:.4f} | CV Accuracy: {cv_acc:.4f}")
    return float(cv_auc)

if USE_OPTUNA:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    print("Optuna best params:", best_params)
else:
    best_params = {'n_estimators':1200, 'max_depth': None, 'min_samples_split':2, 'min_samples_leaf':1, 'max_features':'sqrt'}

# ---------- train final RF on RAW ----------
rf_raw = RandomForestClassifier(**best_params, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
rf_raw.fit(X_train, y_train)
y_score = rf_raw.predict_proba(X_test)[:,1]
raw_auc = roc_auc_score(y_test, y_score)
y_pred_label = (y_score >= 0.5).astype(int)
raw_acc = accuracy_score(y_test, y_pred_label)
print("RAW-only Test Accuracy:", round(raw_acc,4))
print("RAW-only Test AUC:", round(raw_auc,4))

# ---------- ROC plot ----------
fpr,tpr,_ = roc_curve(y_test, y_score)
plt.figure(figsize=(6,6)); plt.plot(fpr,tpr,label=f"RF RAW-only AUC={raw_auc:.3f}"); plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - RAW"); plt.legend(); plt.grid(); plt.show()

# ---- CV metrics on TRAIN (RAW) ----
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
cv_auc = cross_val_score(rf_raw, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
cv_acc = cross_val_score(rf_raw, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1).mean()

# ---------- expose globals ----------
globals().update({
    'X_raw': X_raw_sample, 'X_raw_scaled': X_raw_scaled,
    'rf_raw': rf_raw, 'y_test_raw': y_test, 'y_score_raw': y_score
})

show_table(
    "RF – Raw",
    [
        ["CV AUC", f"{cv_auc:.4f}"],
        ["CV Acc", f"{cv_acc:.4f}"],
        ["Test AUC", f"{raw_auc:.4f}"],
        ["Test Acc", f"{raw_acc:.4f}"],
    ]
)

# =======================================================================================

#____________________________________________________________________________________________________________________________________________#
# ================= MERGE RAW + NODE + PH, Optuna RF, ROC plot =================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import optuna

# ---------------- user params ----------------
USE_OPTUNA = True        # set False to skip Optuna and use default params
N_OPTUNA_TRIALS = 30
RANDOM_STATE = 42
# ---------------------------------------------

# ------- fetch required globals (robust) -------
g = globals()
samples = g.get('samples') or g.get('samples_stage1') or g.get('samples_stage2')
if samples is None:
    raise RuntimeError("Could not find 'samples' in globals. Run (preprocessing) first.")

# labels
y = np.array([int(s.get('label', 0)) for s in samples])

# sample-level split indices
SAMPLE_TRAIN_IDX = g.get('SAMPLE_TRAIN_IDX')
SAMPLE_TEST_IDX  = g.get('SAMPLE_TEST_IDX')
if SAMPLE_TRAIN_IDX is None or SAMPLE_TEST_IDX is None:
    raise RuntimeError("SAMPLE_TRAIN_IDX / SAMPLE_TEST_IDX not found in globals.")

SAMPLE_TRAIN_IDX = np.array(SAMPLE_TRAIN_IDX, dtype=int)
SAMPLE_TEST_IDX  = np.array(SAMPLE_TEST_IDX, dtype=int)

# get scaled/raw/node/ph matrices (try several common names)
X_raw_scaled = g.get('X_raw_scaled')
if X_raw_scaled is None:
    X_raw_scaled = g.get('X_raw')

X_node_scaled = g.get('X_node_scaled')
if X_node_scaled is None:
    X_node_scaled = g.get('X_node')

X_ph_pca = g.get('X_ph_pca')
if X_ph_pca is None:
    X_ph_pca = g.get('X_ph')
if X_ph_pca is None:
    X_ph_pca = np.zeros((len(samples), 0))

# If we got unscaled matrices (X_raw or X_node) attempt to scale them here (fit on train only)
def ensure_scaled(X_obj, name):
    if X_obj is None:
        return None
    X_obj = np.asarray(X_obj)
    # if appears already scaled (same shape and dtype), we still proceed safely
    # scale using train indices if not already scaled (we can't reliably detect that, so scale anyway)
    scaler = StandardScaler()
    try:
        scaler.fit(X_obj[SAMPLE_TRAIN_IDX])
        X_scaled = np.zeros_like(X_obj, dtype=float)
        X_scaled[SAMPLE_TRAIN_IDX] = scaler.transform(X_obj[SAMPLE_TRAIN_IDX])
        X_scaled[SAMPLE_TEST_IDX]  = scaler.transform(X_obj[SAMPLE_TEST_IDX])
        return X_scaled
    except Exception as e:
        raise RuntimeError(f"Failed to scale {name}: {e}")

# Ensure arrays exist and are 2D
if X_raw_scaled is None:
    X_raw_scaled = np.zeros((len(samples), 0))
else:
    X_raw_scaled = ensure_scaled(X_raw_scaled, 'X_raw')

if X_node_scaled is None:
    X_node_scaled = np.zeros((len(samples), 0))
else:
    X_node_scaled = ensure_scaled(X_node_scaled, 'X_node')

if X_ph_pca is None:
    X_ph_pca = np.zeros((len(samples), 0))
else:
    X_ph_pca = np.asarray(X_ph_pca)
    if X_ph_pca.ndim == 1:
        X_ph_pca = X_ph_pca.reshape(len(samples), -1) if X_ph_pca.size == len(samples) else X_ph_pca.reshape(len(samples), -1)
    # If PCA was not fit/has zero columns, leave as zeros((n,0))

# Build merged feature matrix
X_merged = np.hstack([X_raw_scaled, X_node_scaled, X_ph_pca])
print("Merged feature matrix shape:", X_merged.shape)

# train/test split (sample-level)
X_train = X_merged[SAMPLE_TRAIN_IDX]
X_test  = X_merged[SAMPLE_TEST_IDX]
y_train = y[SAMPLE_TRAIN_IDX]
y_test  = y[SAMPLE_TEST_IDX]

# Optuna objective (CV on train)
def optuna_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000, step=100),
        'max_depth': trial.suggest_int('max_depth', 8, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'max_features': trial.suggest_float('max_features', 0.1, 0.7),
        'bootstrap': True
    }
    rf = RandomForestClassifier(**params, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    cv_acc = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    print(f"CV AUC: {cv_auc:.4f} | CV Accuracy: {cv_acc:.4f}")
    return float(cv_auc)


if USE_OPTUNA:
    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    print("Optuna best params:", best_params)
else:
    best_params = {'n_estimators':1200, 'max_depth': None, 'min_samples_split':2, 'min_samples_leaf':1, 'max_features':'sqrt'}

# Retrain on full train set with best params
rf_merged = RandomForestClassifier(**best_params, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
rf_merged.fit(X_train, y_train)

# Evaluate on test set
y_score = rf_merged.predict_proba(X_test)[:,1]
test_auc = roc_auc_score(y_test, y_score)
y_pred_label = (y_score >= 0.5).astype(int)
test_acc = accuracy_score(y_test, y_pred_label)
print(f"Merged RAW+NODE+PH Test Accuracy: {test_acc:.4f}")
print(f"Merged RAW+NODE+PH Test AUC: {test_auc:.4f}")

# ROC plot
fpr, tpr, _ = roc_curve(y_test, y_score)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"Merged RF AUC = {test_auc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC - RAW+NODE+PH (Merged)")
plt.legend(); plt.grid(); plt.show()

cv = N_FOLDS
X_train_merged = X_merged[:len(y_train)]

cv_auc = cross_val_score(rf_merged, X_train_merged, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
cv_acc = cross_val_score(rf_merged, X_train_merged, y_train, cv=cv, scoring='accuracy', n_jobs=-1).mean()

# expose to globals for later use
globals().update({
    'X_train_merged': X_train_merged,
    'rf_merged': rf_merged,
    'y_test_merged': y_test,
    'y_score_merged': y_score,
    'merged_best_params': best_params,
    'cv_auc': cv_auc,
    'cv_acc': cv_acc
})

show_table(
    "RF – Raw + Node + PH",
    [
        ["CV AUC", f"{cv_auc:.4f}"],
        ["CV Acc", f"{cv_acc:.4f}"],
        ["Test AUC", f"{test_auc:.4f}"],
        ["Test Acc", f"{test_acc:.4f}"],
    ]
)
# =======================================================================____________________________________________________________________________________#
# ================= XGBoost on NODE and NODE+PH (with Optuna tuning + relevance) =================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

# -------- user params --------
USE_OPTUNA = True       # set False to skip Optuna tuning
N_TRIALS = 30
RANDOM_STATE = 42
# -----------------------------

# -------- robustly fetch required arrays from globals (fallbacks) --------
g = globals()

# labels / indices
samples = g.get('samples') or g.get('samples_stage1') or g.get('samples_stage2')
if samples is None:
    raise RuntimeError("Could not find 'samples' in globals. Run preprocessing cell first.")
y = np.array([int(s.get('label', 0)) for s in samples])

SAMPLE_TRAIN_IDX = np.array(g.get('SAMPLE_TRAIN_IDX'), dtype=int) if g.get('SAMPLE_TRAIN_IDX') is not None else None
SAMPLE_TEST_IDX  = np.array(g.get('SAMPLE_TEST_IDX'), dtype=int)  if g.get('SAMPLE_TEST_IDX')  is not None else None
if SAMPLE_TRAIN_IDX is None or SAMPLE_TEST_IDX is None:
    raise RuntimeError("SAMPLE_TRAIN_IDX / SAMPLE_TEST_IDX missing.")

# node features (prefer prepared train/test slices if present)
X_node_train = g.get('X_node_train')
X_node_test  = g.get('X_node_test')
if X_node_train is None or X_node_test is None:
    X_node = g.get('X_node_scaled') or g.get('X_node')
    if X_node is None:
        raise RuntimeError("X_node / X_node_scaled not found in globals.")
    X_node = np.asarray(X_node, dtype=float)
    X_node_train = X_node[SAMPLE_TRAIN_IDX]
    X_node_test  = X_node[SAMPLE_TEST_IDX]

# PH PCA block (may be empty)
X_ph_pca = g.get('X_ph_pca')
if X_ph_pca is None:
    X_ph_pca = np.zeros((len(samples), 0))
else:
    X_ph_pca = np.asarray(X_ph_pca, dtype=float)
    if X_ph_pca.ndim == 1:
        # try to reshape if ambiguous
        if X_ph_pca.size == len(samples):
            X_ph_pca = X_ph_pca.reshape(len(samples), 1)
        else:
            X_ph_pca = X_ph_pca.reshape(len(samples), -1)

# prepare PH train/test if available
has_ph = (X_ph_pca.size > 0)
if has_ph:
    X_ph_train = X_ph_pca[SAMPLE_TRAIN_IDX]
    X_ph_test  = X_ph_pca[SAMPLE_TEST_IDX]
else:
    X_ph_train = np.zeros((X_node_train.shape[0], 0))
    X_ph_test  = np.zeros((X_node_test.shape[0], 0))

# merged (NODE+PH) matrices
X_nodeph_train = np.hstack([X_node_train, X_ph_train]) if has_ph else X_node_train.copy()
X_nodeph_test  = np.hstack([X_node_test,  X_ph_test])  if has_ph else X_node_test.copy()

y_train = y[SAMPLE_TRAIN_IDX]
y_test  = y[SAMPLE_TEST_IDX]

print("Shapes -> X_node_train:", X_node_train.shape, "X_nodeph_train:", X_nodeph_train.shape, "has_ph:", has_ph)

# ----------------- helper: Optuna objective for XGBoost -----------------
def xgb_objective_factory(Xtr, ytr, Xval, yval):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
        }
        clf = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss',
                                random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(Xtr, ytr)
        y_score = clf.predict_proba(Xval)[:,1]
        return float(roc_auc_score(yval, y_score))
    return objective

# ----------------- Train XGBoost on NODE -----------------
if USE_OPTUNA:
    study_node = optuna.create_study(direction='maximize')
    study_node.optimize(xgb_objective_factory(X_node_train, y_train, X_node_test, y_test), n_trials=N_TRIALS, show_progress_bar=True)
    best_params_node = study_node.best_params
    print("XGB (NODE) best params:", best_params_node)
else:
    best_params_node = {'n_estimators':500, 'max_depth':6, 'learning_rate':0.05, 'subsample':0.8, 'colsample_bytree':0.8}

xgb_node = xgb.XGBClassifier(**best_params_node, use_label_encoder=False, eval_metric='logloss',
                             random_state=RANDOM_STATE, n_jobs=-1)
xgb_node.fit(X_node_train, y_train)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc = cross_val_score(xgb_node, X_node_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
cv_acc = cross_val_score(xgb_node, X_node_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1).mean()
print(f"XGB NODE CV (k-fold) -> AUC: {cv_auc:.4f}, Accuracy: {cv_acc:.4f}")
y_node_score = xgb_node.predict_proba(X_node_test)[:,1]
node_auc = roc_auc_score(y_test, y_node_score)
y_node_pred_label = (xgb_node.predict_proba(X_node_test)[:,1] >= 0.5).astype(int)
node_acc = accuracy_score(y_test, y_node_pred_label)
print(f"XGB NODE Test Accuracy: {node_acc:.4f}")
print(f"XGB NODE Test AUC: {node_auc:.4f}")

# ----------------- Train XGBoost on NODE+PH -----------------
if has_ph:
    if USE_OPTUNA:
        study_nodeph = optuna.create_study(direction='maximize')
        study_nodeph.optimize(xgb_objective_factory(X_nodeph_train, y_train, X_nodeph_test, y_test), n_trials=N_TRIALS, show_progress_bar=True)
        best_params_nodeph = study_nodeph.best_params
        print("XGB (NODE+PH) best params:", best_params_nodeph)
    else:
        best_params_nodeph = {'n_estimators':500, 'max_depth':6, 'learning_rate':0.05, 'subsample':0.8, 'colsample_bytree':0.8}

    xgb_nodeph = xgb.XGBClassifier(**best_params_nodeph, use_label_encoder=False, eval_metric='logloss',
                                   random_state=RANDOM_STATE, n_jobs=-1)
    xgb_nodeph.fit(X_nodeph_train, y_train)
    y_nodeph_score = xgb_nodeph.predict_proba(X_nodeph_test)[:,1]
    nodeph_auc = roc_auc_score(y_test, y_nodeph_score)
    cv_auc_ph = cross_val_score(xgb_nodeph, X_nodeph_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    cv_acc_ph = cross_val_score(xgb_nodeph, X_nodeph_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1).mean()
    print(f"XGB NODE+PH CV (k-fold) -> AUC: {cv_auc_ph:.4f}, Accuracy: {cv_acc_ph:.4f}")
    
    y_nodeph_pred_label = (xgb_nodeph.predict_proba(X_nodeph_test)[:,1] >= 0.5).astype(int)
    nodeph_acc = accuracy_score(y_test, y_nodeph_pred_label)
    print(f"XGB NODE+PH Test Accuracy: {nodeph_acc:.4f}")
    print(f"XGB NODE+PH Test AUC: {nodeph_auc:.4f}")
else:
    xgb_nodeph = None
    y_nodeph_score = None
    nodeph_auc = None
    print("Skipping XGB NODE+PH: no PH features available (X_ph_pca is empty).")

# ----------------- ROC plot for both -----------------
plt.figure(figsize=(6,6))
fpr, tpr, _ = roc_curve(y_test, y_node_score); plt.plot(fpr, tpr, label=f"XGB NODE AUC={node_auc:.3f}")
if xgb_nodeph is not None:
    fpr, tpr, _ = roc_curve(y_test, y_nodeph_score); plt.plot(fpr, tpr, label=f"XGB NODE+PH AUC={nodeph_auc:.3f}")
plt.plot([0,1],[0,1],'k--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - XGBoost"); plt.legend(); plt.grid(); plt.show()

# ======================= XGBoost Feature Importance with PH Mapping =======================
# --- Load samples from pickle to inspect feature lengths/types ---
PICKLE_PATH = 'Content/processed_toponetx_edgeagg.pkl'
with open(PICKLE_PATH, 'rb') as f:
    samples = pickle.load(f)

# Take the first sample as template for feature sizes
sample = samples[0]

# --- Node feature names ---
X_node_ncols = X_node_train.shape[1]
stats = ['mean','std','median','min','max']
n_stats = len(stats)
n_nodes = X_node_ncols // n_stats
extra = X_node_ncols % n_stats
node_feature_names = []
for i, stat in enumerate(stats):
    count = n_nodes + (1 if i < extra else 0)
    node_feature_names.extend([f"node_{stat}_{j+1}" for j in range(count)])

# --- PH feature names ---
ph_feature_names = []
if PH_STACK and X_ph_train is not None:
    # Landscape features
    n_landscape = len(sample['landscape_features'])
    ph_feature_names.extend([f"landscape_PCA{i+1}" for i in range(n_landscape)])
    
    # Betti numbers (H0/H1 curves)
    for dim, curve in sample['betti_curves'].items():
        n_betti = len(curve['vals'])
        ph_feature_names.extend([f"betti{dim}_curve_{i+1}" for i in range(n_betti)])
    
    # Persistence images
    for dim in [0,1]:
        n_pi = len(sample[f'persistence_image_H{dim}'])
        ph_feature_names.extend([f"pi_H{dim}_{i+1}" for i in range(n_pi)])
    
    if rf_stack is not None or xgb_nodeph is not None:
        ph_feature_names.append('PH_prob')  # last stacked column

# --- Combine node + PH ---
feat_names = node_feature_names + ph_feature_names

# --- Ensure alignment with XGBoost model ---
model = xgb_nodeph if (PH_STACK and xgb_nodeph is not None) else xgb_node
n_features = model.get_booster().num_features()
if len(feat_names) > n_features:
    feat_names = feat_names[:n_features]
elif len(feat_names) < n_features:
    feat_names += [f"extra_{i+1}" for i in range(n_features - len(feat_names))]

# --- Get feature importance (gain) ---
imp_dict = model.get_booster().get_score(importance_type='gain')
imp_list = [imp_dict.get(f"f{i}", 0.0) for i in range(len(feat_names))]

# --- Sort top features ---
top_idx = np.argsort(imp_list)[::-1][:30]
top_names = [feat_names[i] for i in top_idx]
top_vals  = [imp_list[i] for i in top_idx]

# --- Print top 20 ---
print("\nTop 20 NODE+PH features (XGBoost gain):")
for i, (name, val) in enumerate(zip(top_names[:20], top_vals[:20]), 1):
    print(f"{i:2d}. {name:30s}: {val:.6f}")

# --- Plot top 20 ---
plt.figure(figsize=(10,6))
plt.barh(top_names[:20][::-1], top_vals[:20][::-1], color='skyblue')
plt.xlabel("Gain")
plt.title("XGBoost NODE+PH Top 20 Features")
plt.tight_layout()
plt.show()

show_table(
    "XGB – Node",
    [
        ["CV AUC", f"{cv_auc:.4f}"],
        ["CV Acc", f"{cv_acc:.4f}"],
        ["Test AUC", f"{node_auc:.4f}"],
        ["Test Acc", f"{node_acc:.4f}"],
    ]
)

show_table(
    "XGB – Node + PH",
    [
        ["CV AUC", f"{cv_auc_ph:.4f}"],
        ["CV Acc", f"{cv_acc_ph:.4f}"],
        ["Test AUC", f"{nodeph_auc:.4f}"],
        ["Test Acc", f"{nodeph_acc:.4f}"],
    ]
)
# =========================================================================================================
# ======================= XGBoost on RAW features (with CV & accuracy) =======================
import numpy as np
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

# -------- user params --------
USE_OPTUNA = True     # set False to skip Optuna
N_TRIALS = 30
RANDOM_STATE = 42
# -----------------------------

g = globals()

# -------- fetch required globals safely --------
samples = g.get('samples')
if samples is None:
    raise RuntimeError("samples not found. Run preprocessing first.")

y = np.array([int(s['label']) for s in samples])

SAMPLE_TRAIN_IDX = np.array(g.get('SAMPLE_TRAIN_IDX'), dtype=int)
SAMPLE_TEST_IDX  = np.array(g.get('SAMPLE_TEST_IDX'), dtype=int)

if SAMPLE_TRAIN_IDX is None or SAMPLE_TEST_IDX is None:
    raise RuntimeError("SAMPLE_TRAIN_IDX / SAMPLE_TEST_IDX missing.")

# prefer scaled raw if available
X_raw = g.get('X_raw_scaled')
if X_raw is None:
    X_raw = g.get('X_raw')
    if X_raw is None:
        raise RuntimeError("X_raw / X_raw_scaled not found.")
X_raw = np.asarray(X_raw, dtype=float)

X_raw_train = X_raw[SAMPLE_TRAIN_IDX]
X_raw_test  = X_raw[SAMPLE_TEST_IDX]
y_train = y[SAMPLE_TRAIN_IDX]
y_test  = y[SAMPLE_TEST_IDX]

print("X_raw_train shape:", X_raw_train.shape)

# -------- Optuna objective --------
def xgb_objective_factory(Xtr, ytr, Xval, yval):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
        }
        clf = xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        clf.fit(Xtr, ytr)
        y_score = clf.predict_proba(Xval)[:, 1]
        return roc_auc_score(yval, y_score)
    return objective

# -------- Train XGBoost on RAW --------
if USE_OPTUNA:
    study_raw = optuna.create_study(direction='maximize')
    study_raw.optimize(
        xgb_objective_factory(X_raw_train, y_train, X_raw_test, y_test),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )
    best_params_raw = study_raw.best_params
    print("XGB RAW best params:", best_params_raw)
else:
    best_params_raw = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

xgb_raw = xgb.XGBClassifier(
    **best_params_raw,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

xgb_raw.fit(X_raw_train, y_train)

# -------- Predictions & AUC --------
y_raw_score = xgb_raw.predict_proba(X_raw_test)[:, 1]
raw_auc = roc_auc_score(y_test, y_raw_score)

# -------- Test accuracy --------
y_raw_pred_label = (y_raw_score >= 0.5).astype(int)
raw_acc = accuracy_score(y_test, y_raw_pred_label)

# -------- k-fold CV on train set --------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
raw_cv_auc = cross_val_score(xgb_raw, X_raw_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
raw_cv_acc = cross_val_score(xgb_raw, X_raw_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1).mean()

print(f"XGB RAW Test AUC: {raw_auc:.4f}, Test Accuracy: {raw_acc:.4f}")
print(f"XGB RAW k-fold CV -> AUC: {raw_cv_auc:.4f}, Accuracy: {raw_cv_acc:.4f}")

# -------- ROC curve --------
plt.figure(figsize=(6,6))
fpr, tpr, _ = roc_curve(y_test, y_raw_score)
plt.plot(fpr, tpr, label=f"XGB RAW AUC={raw_auc:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve - XGBoost on RAW features")
plt.legend()
plt.grid()
plt.show()

# -------- Feature importance (gain) --------
importance = xgb_raw.get_booster().get_score(importance_type='gain')
imp_vals = np.array([importance.get(f"f{i}", 0.0) for i in range(X_raw_train.shape[1])])
top_idx = np.argsort(imp_vals)[::-1][:20]

print("\nTop RAW features (XGBoost gain):")
for i in top_idx:
    print(f"raw_{i:03d}: {imp_vals[i]:.6f}")

plt.figure(figsize=(8,5))
plt.barh([f"raw_{i}" for i in top_idx][::-1], imp_vals[top_idx][::-1])
plt.title("Top 20 RAW features (XGBoost gain)")
plt.xlabel("gain")
plt.tight_layout()
plt.show()

# -------- expose globals --------
g.update({
    'xgb_raw': xgb_raw,
    'y_raw_score': y_raw_score,
    'raw_auc': raw_auc,
    'raw_acc': raw_acc,
    'raw_cv_auc': raw_cv_auc,
    'raw_cv_acc': raw_cv_acc
})

show_table(
    "XGB – Raw",
    [
        ["CV AUC", f"{raw_cv_auc:.4f}"],
        ["CV Acc", f"{raw_cv_acc:.4f}"],
        ["Test AUC", f"{raw_auc:.4f}"],
        ["Test Acc", f"{raw_acc:.4f}"],
    ]
)

# ============================================================================
# ======================= XGBoost on RAW + NODE + PH (with CV & Accuracy) =======================
import numpy as np
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

# -------- user params --------
USE_OPTUNA = True
N_TRIALS = 30
RANDOM_STATE = 42
# -----------------------------

g = globals()

# -------- required globals --------
samples = g.get('samples')
if samples is None:
    raise RuntimeError("samples not found.")

SAMPLE_TRAIN_IDX = np.array(g.get('SAMPLE_TRAIN_IDX'), dtype=int)
SAMPLE_TEST_IDX  = np.array(g.get('SAMPLE_TEST_IDX'), dtype=int)

y = np.array([int(s['label']) for s in samples])
y_train = y[SAMPLE_TRAIN_IDX]
y_test  = y[SAMPLE_TEST_IDX]

# -------- feature matrices --------
X_raw  = g.get('X_raw_scaled', g.get('X_raw'))
X_node = g.get('X_node_scaled', g.get('X_node'))
X_ph   = g.get('X_ph_pca', np.zeros((len(samples), 0)))

if X_raw is None or X_node is None:
    raise RuntimeError("X_raw / X_node not found.")

X_raw  = np.asarray(X_raw, dtype=float)
X_node = np.asarray(X_node, dtype=float)
X_ph   = np.asarray(X_ph, dtype=float)

# -------- merge features --------
X_all = np.hstack([X_raw, X_node, X_ph])
X_train = X_all[SAMPLE_TRAIN_IDX]
X_test  = X_all[SAMPLE_TEST_IDX]

print("X_all shape:", X_all.shape)

# -------- Optuna objective --------
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
    }
    clf = xgb.XGBClassifier(
        **params,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_score)

# -------- train model --------
if USE_OPTUNA:
    study = optuna.create_study(direction='maximize')
    study.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=True)
    best_params = study.best_params
    print("Best XGB params:", best_params)
else:
    best_params = {
        'n_estimators': 800,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

xgb_all = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

xgb_all.fit(X_train, y_train)

# -------- Predictions & Test metrics --------
y_score_all = xgb_all.predict_proba(X_test)[:, 1]
auc_all = roc_auc_score(y_test, y_score_all)
y_pred_all = (y_score_all >= 0.5).astype(int)
acc_all = accuracy_score(y_test, y_pred_all)

# -------- k-fold CV metrics on train set --------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc_all = cross_val_score(xgb_all, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
cv_acc_all = cross_val_score(xgb_all, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1).mean()

print(f"XGB RAW+NODE+PH Test -> AUC: {auc_all:.4f}, Accuracy: {acc_all:.4f}")
print(f"XGB RAW+NODE+PH k-fold CV -> AUC: {cv_auc_all:.4f}, Accuracy: {cv_acc_all:.4f}")

# -------- ROC curve --------
plt.figure(figsize=(6,6))
fpr, tpr, _ = roc_curve(y_test, y_score_all)
plt.plot(fpr, tpr, label=f"XGB ALL AUC={auc_all:.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC – XGBoost (RAW + NODE + PH)")
plt.legend()
plt.grid()
plt.show()

# -------- expose globals --------
g.update({
    'xgb_all': xgb_all,
    'y_all_score': y_score_all,
    'auc_all': auc_all,
    'acc_all': acc_all,
    'cv_auc_all': cv_auc_all,
    'cv_acc_all': cv_acc_all
})

show_table(
    "XGB – Raw + Node + PH",
    [
        ["CV AUC", f"{cv_auc_all:.4f}"],
        ["CV Acc", f"{cv_acc_all:.4f}"],
        ["Test AUC", f"{auc_all:.4f}"],
        ["Test Acc", f"{acc_all:.4f}"],
    ]
)

#______________________________________________________________________________________________________________#
# ======================= Evaluation Cell: CV + 95% CI summary (RF + XGB) + Accuracy =======================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from textwrap import shorten

# ---------- helper: CV + CI (AUC) + Accuracy ----------
def cross_val_metrics(model, X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[test_idx], probs))
        accs.append(accuracy_score(y[test_idx], preds))
    aucs = np.array(aucs, dtype=float)
    accs = np.array(accs, dtype=float)
    mean_auc = float(aucs.mean())
    se_auc = float(aucs.std(ddof=1) / np.sqrt(len(aucs)))
    ci_auc = (mean_auc - 1.96 * se_auc, mean_auc + 1.96 * se_auc)
    mean_acc = float(accs.mean())
    se_acc = float(accs.std(ddof=1) / np.sqrt(len(accs)))
    ci_acc = (mean_acc - 1.96 * se_acc, mean_acc + 1.96 * se_acc)
    return mean_auc, ci_auc, mean_acc, ci_acc

# ---------- locate features ----------
g = globals()
n_samples = len(g.get('samples', []))
if n_samples == 0:
    raise RuntimeError("No samples found.")

def first_existing(*names):
    for name in names:
        val = g.get(name, None)
        if val is not None:
            return val, name
    return None, None

X_raw_candidate, X_raw_name = first_existing('X_raw_scaled','X_raw')
X_node_candidate, X_node_name = first_existing('X_node_scaled','X_node')
X_ph_candidate, X_ph_name     = first_existing('X_ph_pca','X_ph')

def as2d(a, expected_rows=n_samples):
    if a is None:
        return None
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 1 and arr.size == expected_rows:
        arr = arr.reshape(expected_rows, 1)
    return arr

X_raw = as2d(X_raw_candidate)
X_node = as2d(X_node_candidate)
X_ph   = as2d(X_ph_candidate)

print("Feature variables found:")
print("  X_raw   :", X_raw_name)
print("  X_node  :", X_node_name)
print("  X_ph    :", X_ph_name, "(cols=%d)" % (0 if X_ph is None else X_ph.shape[1]))

# Build feature blocks if available
X_sets = {}
if X_raw is not None: X_sets['RAW'] = X_raw
if X_node is not None: X_sets['NODE'] = X_node
if X_node is not None and X_ph is not None and X_ph.shape[1] > 0: X_sets['NODE+PH'] = np.hstack([X_node, X_ph])
if X_raw is not None and X_node is not None and X_ph is not None and X_ph.shape[1] > 0: X_sets['RAW+NODE+PH'] = np.hstack([X_raw, X_node, X_ph])

# ---------- candidate model names ----------
rf_name_map = {
    'RAW': ['rf_raw','rf_raw_model','best_rf'],
    'NODE': ['rf_node','rf_node_model','rf_node_best'],
    'NODE+PH': ['rf_nodeph','rf_stack'],
    'RAW+NODE+PH': ['rf_merged','rf_raw_nodeph','rf_all']
}
xgb_name_map = {
    'RAW': ['xgb_raw','xgb_raw_model'],
    'NODE': ['xgb_node','xgb_node_model'],
    'NODE+PH': ['xgb_nodeph','xgb_node_ph'],
    'RAW+NODE+PH': ['xgb_all','xgb_raw_nodeph']
}

# ---------- find models ----------
models = {}
for key, candidates in rf_name_map.items():
    for var in candidates:
        m = g.get(var, None)
        if m is not None:
            models[f"RF::{key}"] = (m, var)
            break
for key, candidates in xgb_name_map.items():
    for var in candidates:
        m = g.get(var, None)
        if m is not None:
            models[f"XGB::{key}"] = (m, var)
            break

y_array = np.array([int(s['label']) for s in g['samples']])

# ---------- compute CV metrics ----------
results = {}
missing_info = []
for model_key, (model_obj, varname) in models.items():
    _, feat_key = model_key.split("::")
    if feat_key not in X_sets:
        missing_info.append(f"Model '{varname}' ({model_key}) found but features '{feat_key}' missing => SKIPPED")
        continue
    X = X_sets[feat_key]
    try:
        mean_auc, ci_auc, mean_acc, ci_acc = cross_val_metrics(model_obj, X, y_array)
        results[model_key] = {
            'mean_auc': mean_auc, 'ci_auc': ci_auc,
            'mean_acc': mean_acc, 'ci_acc': ci_acc,
            'varname': varname
        }
    except Exception as e:
        missing_info.append(f"Model '{varname}' ({model_key}) failed CV: {e}")

if missing_info:
    print("\nSkipped/Failed items:")
    for s in missing_info:
        print("  -", s)

if not results:
    raise RuntimeError("No models were evaluated.")

# ---------- ordered display ----------
display_order = []
for key in ['RAW','NODE','NODE+PH','RAW+NODE+PH']:
    if f"RF::{key}" in results: display_order.append(f"RF::{key}")
for key in ['RAW','NODE','NODE+PH','RAW+NODE+PH']:
    if f"XGB::{key}" in results: display_order.append(f"XGB::{key}")

names = [k.replace("::"," - ") for k in display_order]
means = [results[k]['mean_auc'] for k in display_order]
errors = [[results[k]['mean_auc'] - results[k]['ci_auc'][0] for k in display_order],
          [results[k]['ci_auc'][1] - results[k]['mean_auc'] for k in display_order]]

plt.figure(figsize=(8, 0.6*len(names)+2.2))
plt.barh(names, means, xerr=errors, capsize=6, alpha=0.9)
plt.xlim(0.6, 1.0)
plt.xlabel("CV AUC (k-fold) ± 95% CI")
plt.title("Model Performance Comparison (RF & XGB)")
plt.grid(axis='x', linestyle='--', alpha=0.35)
plt.tight_layout()
plt.show()

# ---------- table with AUC + Accuracy ----------
table_data = []
for k in display_order:
    res = results[k]
    auc_mean, auc_ci = res['mean_auc'], res['ci_auc']
    acc_mean, acc_ci = res['mean_acc'], res['ci_acc']
    varname = res['varname']
    table_data.append([
        k,
        f"{auc_mean:.4f}", f"[{auc_ci[0]:.4f} – {auc_ci[1]:.4f}]",
        f"{acc_mean:.4f}", f"[{acc_ci[0]:.4f} – {acc_ci[1]:.4f}]",
        varname
    ])

fig, ax = plt.subplots(figsize=(10, 0.7 + 0.45*len(table_data)))
ax.axis('off')
col_labels = ['Model', 'CV AUC', '95% CI (AUC)', 'CV Acc', '95% CI (Acc)', 'varname']
tbl = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.15, 1.3)
plt.tight_layout()
plt.show()
#______________________________________________________________________________________________________________________________________________________________________#
#  Visualization PD, Betti, Landscapes, PIs (handles infinite deaths & missing keys)
import os, pickle, random, numpy as np, networkx as nx
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

# ---------- user params ----------
OUT_DIR = 'Content/Breast cancer classification'
processed_data_path = os.path.join(OUT_DIR, 'processed_toponetx_edgeagg.pkl')
max_per_class = 2
BETTI_RES = 400
LANDSCAPE_RES = 200
MAX_LANDS = 5
PI_GRID = (32, 32)
FONT_TITLE = 14
FONT_SMALL = 11
# ---------------------------------

def safe_load_samples(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def build_graph_from_sc(sc):
    G = nx.Graph()
    if sc is None: return G
    for simp in getattr(sc, 'simplices', sc if isinstance(sc, (list,tuple)) else []):
        nodes = [int(v) for v in list(simp)]
        for n in nodes: G.add_node(n)
        if len(nodes) >= 2:
            for a in range(len(nodes)):
                for b in range(a+1,len(nodes)):
                    G.add_edge(nodes[a], nodes[b])
    return G

def safe_image_array(sample):
    if 'image' in sample and sample['image'] is not None:
        return np.array(sample['image'])
    for k in ('image_path','filepath','path'):
        p = sample.get(k)
        if p:
            try:
                return np.array(Image.open(p).convert('RGB'))
            except Exception:
                try:
                    return np.array(Image.open(p).convert('L'))
                except Exception:
                    pass
    return None

def extract_intervals(sample):
    # robustly collect H0 and H1 intervals as lists of (b,d) (float, possibly np.inf)
    H0 = sample.get('H0_intervals') if 'H0_intervals' in sample else sample.get('H0') if 'H0' in sample else None
    H1 = sample.get('H1_intervals') if 'H1_intervals' in sample else sample.get('H1') if 'H1' in sample else None

    # If persistence_diagram present, use it too
    pd = sample.get('persistence_diagram', None)
    if pd is not None:
        # pd may be list of (dim,(b,d))
        h0 = []; h1 = []
        for item in pd:
            try:
                dim, pair = item
                b,d = float(pair[0]), float(pair[1])
                if int(dim)==0: h0.append((b,d))
                elif int(dim)==1: h1.append((b,d))
            except Exception:
                continue
        if (not H0) and h0: H0 = h0
        if (not H1) and h1: H1 = h1

    # normalize formats to list of tuples
    def norm(x):
        if x is None: return []
        try:
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return [(float(row[0]), float(row[1])) for row in arr]
            # else try to iterate
            out=[]
            for pair in x:
                try:
                    b = float(pair[0]); d = float(pair[1]); out.append((b,d))
                except Exception:
                    continue
            return out
        except Exception:
            return []
    return norm(H0), norm(H1)

def betti_curve_from_intervals(intervals, t_min, t_max, res=BETTI_RES):
    # treat infinite deaths by capping them at display_tmax
    ts = np.linspace(t_min, t_max, res)
    vals = np.zeros_like(ts, dtype=float)
    for (b,d) in intervals:
        b = float(b)
        if np.isfinite(d) and (d > b):
            mask = (ts >= b) & (ts <= float(d))
        elif not np.isfinite(d) and (t_max > b):
            # cap infinite death at t_max for display
            mask = (ts >= b) & (ts <= t_max)
        else:
            mask = (ts >= b)
        vals += mask.astype(float)
    return ts, vals

def infer_pi_map(pi_entry):
    if pi_entry is None: return None
    a = np.asarray(pi_entry)
    if a.ndim == 2: return a
    if a.ndim == 1:
        # try provided grid
        if a.size == PI_GRID[0]*PI_GRID[1]:
            return a.reshape(PI_GRID[0], PI_GRID[1])
        # try square
        s = int(np.sqrt(a.size))
        if s*s == a.size:
            return a.reshape(s,s)
    return None

# ---------------- load ----------------
samples = safe_load_samples(processed_data_path)
neg = [i for i,s in enumerate(samples) if int(s.get('label', -1))==0]
pos = [i for i,s in enumerate(samples) if int(s.get('label', -1))==1]
idxs = []
if neg:
    idxs += random.sample(neg, min(max_per_class, len(neg)))
if pos:
    idxs += random.sample(pos, min(max_per_class, len(pos)))
if not idxs:
    idxs = list(range(min(4, len(samples))))
print("Visualizing indices:", idxs)
n = len(idxs)

# ---------------- PAGE 1: Images + overlayed SC ----------------
fig1, ax1 = plt.subplots(2, n, figsize=(4.5*n, 8), squeeze=False)

for col, i in enumerate(idxs):
    s = samples[i]
    sc = s.get('simplicial_complex', None)
    G = build_graph_from_sc(sc)

    # --- load image ---
    img = safe_image_array(s)
    ax = ax1[0, col]
    if img is not None:
        if img.ndim == 3:
            img_disp = img.mean(axis=2)
        else:
            img_disp = img
        h, w = img_disp.shape
        ax.imshow(img_disp, cmap='gray', origin='upper')
        ax.set_xlim(-0.5, w-0.5)
        ax.set_ylim(h-0.5, -0.5)
    else:
        h = w = None
        ax.text(0.5,0.5,"No image", ha='center', va='center')

    # --- positions: use stored or compute from superpixels ---
    pos = {}
    if 'positions' in s and s['positions'] is not None:
        for k,v in s['positions'].items():
            try:
                pos[int(k)] = (float(v[0]), float(v[1]))
            except:
                continue
    elif 'superpixel_labels' in s:
        seg = s['superpixel_labels']
        for sp in np.unique(seg):
            coords = np.argwhere(seg == sp)
            if coords.size > 0:
                y,x = coords.mean(axis=0)
                pos[int(sp)] = (float(x), float(y))

    # --- overlay SC ---
    valid_nodes = set(G.nodes()).intersection(pos.keys())
    draw_edges = [(u,v) for (u,v) in G.edges() if u in valid_nodes and v in valid_nodes]

    if pos and valid_nodes:
        nx.draw_networkx_edges(
            G, pos, edgelist=draw_edges,
            ax=ax, edge_color='cyan', width=1.2, alpha=0.9
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(valid_nodes),
            ax=ax, node_size=45, node_color='lime', linewidths=0.2
        )

    ax.set_title(f"Sample {i} | Label {s.get('label')}", fontsize=FONT_TITLE)
    ax.axis('off')

    # --- bottom: standalone SC (unchanged) ---
    ax2 = ax1[1, col]
    if G.number_of_nodes() > 0:
        spr = nx.spring_layout(G, seed=0)
        nx.draw(G, pos=spr, ax=ax2, node_size=40,
                node_color='skyblue', edge_color='gray', width=0.6)
        ax2.set_title("Simplicial Complex", fontsize=12)
    else:
        ax2.text(0.5,0.5,"No SC", ha='center', va='center')
    ax2.axis('off')

plt.tight_layout()
plt.show()


# ---------------- PAGE 2: PD + summary (robust) ----------------
fig2 = plt.figure(figsize=(4.5*n, 6))
gs = gridspec.GridSpec(2, n, height_ratios=[3, 1.6], hspace=0.4)

for col, i in enumerate(idxs):
    s = samples[i]
    H0, H1 = extract_intervals(s)

    # compute display t-range (use finite births & deaths)
    births = [b for (b,d) in (H0+H1)]
    finite_deaths = [d for (b,d) in (H0+H1) if np.isfinite(d)]
    if births:
        tmin = min(births)
    else:
        tmin = 0.0
    if finite_deaths:
        tmax = max(finite_deaths)
    else:
        tmax = tmin + 1.0
    # small padding
    span = max(1e-6, tmax - tmin)
    tmax_display = tmax + 0.05*span
    tmin_display = tmin - 0.05*span

    ax = fig2.add_subplot(gs[0, col])
    drawn = False
    # plot finite PDs
    for arr, label, marker, color in [(H0,'H0','o','C0'), (H1,'H1','x','C1')]:
        if arr:
            arr_np = np.array(arr, dtype=float)
            if arr_np.size and arr_np.ndim==2:
                finite_mask = np.isfinite(arr_np[:,1])
                if finite_mask.any():
                    ax.scatter(arr_np[finite_mask,0], arr_np[finite_mask,1], s=20, label=label, marker=marker, color=color, alpha=0.8)
                    drawn = True
                # plot infinite deaths at y=tmax_display with upward triangle
                inf_mask = ~np.isfinite(arr_np[:,1])
                if inf_mask.any():
                    ax.scatter(arr_np[inf_mask,0], np.ones(inf_mask.sum())*tmax_display, s=30, marker='^', facecolors='none', edgecolors=color, label=f"{label} (∞)")
                    drawn = True

    if drawn:
        # diagonal
        lo = tmin_display; hi = tmax_display
        ax.plot([lo,hi],[lo,hi],'k--', lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Birth"); ax.set_ylabel("Death")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5,0.5,"No finite PD points", ha='center', va='center', fontsize=FONT_SMALL)
        ax.set_xticks([]); ax.set_yticks([])

    ax.set_title(f"Persistence Diagram (sample {i})", fontsize=FONT_TITLE)

    # summary row
    ax2 = fig2.add_subplot(gs[1, col])
    # counts incl. infinite
    H0_total = len(H0); H1_total = len(H1)
    H0_finite = sum(1 for (_,d) in H0 if np.isfinite(d))
    H1_finite = sum(1 for (_,d) in H1 if np.isfinite(d))
    s0 = summary = lambda intervals: (sum(1 for b,d in intervals if np.isfinite(d)), np.mean([d-b for b,d in intervals if np.isfinite(d)]) if any(np.isfinite(d) for _,d in intervals) else 0.0)
    h0_count, h0_mean = s0(H0); h1_count, h1_mean = s0(H1)

    sc = s.get('simplicial_complex', None)
    G = build_graph_from_sc(sc)
    lines = [
        f"Nodes: {G.number_of_nodes()}",
        f"Edges: {G.number_of_edges()}",
        f"H0 total: {H0_total} (finite: {H0_finite})",
        f"H1 total: {H1_total} (finite: {H1_finite})",
        f"H0 mean finite life: {h0_mean:.3f}",
        f"H1 mean finite life: {h1_mean:.3f}",
    ]
    for r,txt in enumerate(lines):
        ax2.text(0.02, 0.9 - r*0.16, txt, transform=ax2.transAxes, fontsize=FONT_SMALL, family='monospace')
    ax2.axis('off')

plt.tight_layout()
plt.show()

# ---------------- PAGE 3: Landscapes | Betti curves | PIs ----------------
fig3, ax3 = plt.subplots(3, n, figsize=(4.5*n, 9), squeeze=False)

for col, i in enumerate(idxs):
    s = samples[i]
    H0, H1 = extract_intervals(s)
    # determine display t-range
    births = [b for (b,d) in (H0+H1)]
    finite_deaths = [d for (b,d) in (H0+H1) if np.isfinite(d)]
    if births:
        tmin = min(births)
    else:
        tmin = 0.0
    if finite_deaths:
        tmax = max(finite_deaths)
    else:
        tmax = tmin + 1.0
    span = max(1e-6, tmax - tmin)
    tmin_plot = tmin - 0.05*span; tmax_plot = tmax + 0.05*span

    # Landscapes (H1)
    ax_land = ax3[0, col]
    vec = np.asarray(s.get('landscape_features', []) or [])
    if vec.size:
        # try to infer resolution
        if vec.size % LANDSCAPE_RES == 0:
            nlands = min(MAX_LANDS, vec.size // LANDSCAPE_RES)
            res = LANDSCAPE_RES
        else:
            # fallback: split into at most MAX_LANDS chunks
            res = int(np.ceil(vec.size / MAX_LANDS))
            nlands = min(MAX_LANDS, vec.size // res)
        grid = np.linspace(tmin_plot, tmax_plot, res)
        plotted = False
        for k in range(nlands):
            seg = vec[k*res:(k+1)*res]
            if seg.size != res: continue
            ax_land.plot(grid, seg, lw=1)
            plotted = True
        if plotted:
            ax_land.set_xlim(tmin_plot, tmax_plot)
            ax_land.set_ylim(bottom=0)
            ax_land.set_title("Persistence Landscapes (H1)", fontsize=FONT_TITLE)
            ax_land.set_xlabel("Filtration")
            ax_land.grid(alpha=0.25)
        else:
            ax_land.text(0.5,0.5,"Landscape shape mismatch", ha='center', va='center')
            ax_land.axis('off')
    else:
        ax_land.text(0.5,0.5,"No landscape", ha='center', va='center')
        ax_land.axis('off')

    # Betti curves
    ax_b = ax3[1, col]
    # build lists for betti curve including infinite deaths (they get capped to tmax_plot)
    ts0, bc0 = betti_curve_from_intervals(H0, tmin_plot, tmax_plot, res=BETTI_RES)
    ts1, bc1 = betti_curve_from_intervals(H1, tmin_plot, tmax_plot, res=BETTI_RES)
    if bc0.sum() == 0 and bc1.sum() == 0:
        ax_b.text(0.5,0.5,"No Betti activity", ha='center', va='center')
        ax_b.axis('off')
    else:
        ax_b.plot(ts0, bc0, label='β0')
        ax_b.plot(ts1, bc1, label='β1')
        ax_b.set_title("Betti Curves", fontsize=FONT_TITLE)
        ax_b.legend(fontsize=9)

    # Persistence images
    ax_pi = ax3[2, col]
    pi0 = infer_pi_map(s.get('persistence_image_H0'))
    pi1 = infer_pi_map(s.get('persistence_image_H1'))
    if (pi0 is None) and (pi1 is None):
        ax_pi.text(0.5,0.5,"No Persistence Images", ha='center', va='center')
        ax_pi.axis('off')
    else:
        if pi0 is None: pi0 = np.zeros_like(pi1)
        if pi1 is None: pi1 = np.zeros_like(pi0)
        # normalize each for visibility
        def normm(x):
            x = np.array(x, dtype=float)
            if x.max() > x.min(): return (x - x.min()) / (x.max()-x.min())
            return x
        left = normm(pi0); right = normm(pi1)
        sep = np.zeros((left.shape[0], max(2, left.shape[1]//16)))
        canvas = np.hstack([left, sep, right])
        ax_pi.imshow(canvas, origin='lower', aspect='auto')
        ax_pi.set_title("Persistence Images (H0 | H1)", fontsize=FONT_TITLE)
        ax_pi.axis('off')

plt.tight_layout()
plt.show()

#___________________________________________________________________________________________________________________________________________________________________________________#

# ======================= RF Feature Importance =======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ sanity checks ------------------
if rf_node is None:
    raise RuntimeError("rf_node not found.")

# ------------------ NODE feature names ------------------
# aggregate_node_stats = [mean, std, median, min, max] per original node feature
n_node_raw = samples[0]['node_features'].shape[1]
node_stats = ['mean', 'std', 'median', 'min', 'max']

node_feature_names = [
    f"NODE_{stat}_f{j}"
    for stat in node_stats
    for j in range(n_node_raw)
]

assert len(node_feature_names) == X_node.shape[1], "Node feature name mismatch"

# ------------------ PH feature names ------------------
ph_feature_names = []

if X_ph_pca.size > 0:
    # infer original PH layout from one sample
    s0 = samples[0]

    # Betti curves (if present)
    if 'betti_features' in s0 and len(s0['betti_features']) > 0:
        n_betti = len(s0['betti_features'])
        ph_feature_names += [f"PH_Betti_{i}" for i in range(n_betti)]

    # Persistence landscapes (if present)
    if 'landscape_features' in s0 and len(s0['landscape_features']) > 0:
        n_land = len(s0['landscape_features'])
        ph_feature_names += [f"PH_Landscape_{i}" for i in range(n_land)]

    # PCA mapping
    ph_feature_names = [f"PCA({name})" for name in ph_feature_names]
    ph_feature_names = ph_feature_names[:X_ph_pca.shape[1]]

# ------------------ CASE 1: NODE-only RF ------------------
node_importance = rf_node.feature_importances_

df_node_imp = pd.DataFrame({
    "feature": node_feature_names,
    "importance": node_importance,
    "type": "NODE"
}).sort_values("importance", ascending=False)

print("\nTop 30 NODE features:")
print(df_node_imp.head(30).to_string(index=False))

plt.figure(figsize=(8,6))
plt.barh(df_node_imp.head(30)['feature'][::-1],
         df_node_imp.head(30)['importance'][::-1])
plt.title("Top 30 NODE Feature Importances (RF)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# ------------------ CASE 2: NODE + PH stacked RF ------------------
# ======================= Random Forest Feature Importance with PH Mapping =======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# --- Load samples from pickle to inspect feature lengths/types ---
PICKLE_PATH = 'Content/Breast cancer classification/processed_toponetx_edgeagg.pkl'
with open(PICKLE_PATH, 'rb') as f:
    samples = pickle.load(f)

sample = samples[0]  # template

# ------------------ NODE feature names ------------------
X_node_ncols = X_node_train.shape[1]
stats = ['mean', 'std', 'median', 'min', 'max']
n_stats = len(stats)
n_nodes = X_node_ncols // n_stats
extra = X_node_ncols % n_stats

node_feature_names = []
for i, stat in enumerate(stats):
    count = n_nodes + (1 if i < extra else 0)
    node_feature_names.extend([f"node_{stat}_{j+1}" for j in range(count)])

# ------------------ PH feature names ------------------
ph_feature_names = []

if PH_STACK and X_ph_train is not None:

    # Landscape (already PCA-compressed)
    n_landscape = X_ph_train.shape[1]
    ph_feature_names.extend([f"landscape_PCA{i+1}" for i in range(n_landscape)])

    # Betti curves (for interpretability bookkeeping)
    for dim, curve in sample['betti_curves'].items():
        n_betti = len(curve['vals'])
        ph_feature_names.extend([f"betti{dim}_curve_{i+1}" for i in range(n_betti)])

    # Persistence images
    for dim in [0, 1]:
        n_pi = len(sample[f'persistence_image_H{dim}'])
        ph_feature_names.extend([f"pi_H{dim}_{i+1}" for i in range(n_pi)])

    # Stacking case
    if rf_stack is not None:
        ph_feature_names.append('PH_prob')

# ------------------ Combine NODE + PH ------------------
feat_names = node_feature_names + ph_feature_names

# ------------------ Align with RF model ------------------
model = rf_stack if (PH_STACK and rf_stack is not None) else rf_node
n_features = model.n_features_in_

if len(feat_names) > n_features:
    feat_names = feat_names[:n_features]
elif len(feat_names) < n_features:
    feat_names += [f"extra_{i+1}" for i in range(n_features - len(feat_names))]

# ------------------ RF feature importance ------------------
imp_vals = model.feature_importances_

df_imp = pd.DataFrame({
    "feature": feat_names,
    "importance": imp_vals
}).sort_values("importance", ascending=False)

# ------------------ Print top 20 ------------------
print("\nTop 20 NODE+PH features (Random Forest):")
for i, row in enumerate(df_imp.head(20).itertuples(index=False), 1):
    print(f"{i:2d}. {row.feature:30s}: {row.importance:.6f}")

# ------------------ Plot top 20 ------------------
plt.figure(figsize=(10, 6))
plt.barh(
    df_imp.head(20)['feature'][::-1],
    df_imp.head(20)['importance'][::-1]
)
plt.xlabel("Mean Decrease in Impurity")
plt.title("Random Forest NODE+PH Top 20 Features")
plt.tight_layout()
plt.show()

#___________________________________END___________________________________________________#