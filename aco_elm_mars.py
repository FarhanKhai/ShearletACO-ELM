import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# 1. DATA LOADING ENGINE
def load_data_fixed(feature_type):
    f_train = f"{feature_type}_train.npz"
    f_val   = f"{feature_type}_val.npz"
    f_test  = f"{feature_type}_test.npz"

    if not (os.path.exists(f_train) and os.path.exists(f_val) and os.path.exists(f_test)):
        return None
    
    train = np.load(f_train)
    val   = np.load(f_val)
    test  = np.load(f_test)
    return (train["X"], train["y"]), (val["X"], val["y"]), (test["X"], test["y"])

def load_data_random(feature_type):
    
    data_fixed = load_data_fixed(feature_type)
    if data_fixed is None: return None
    
    (tr_X, tr_y), (val_X, val_y), (te_X, te_y) = data_fixed
    
    # Gabung jadi data besar
    X_all = np.vstack([tr_X, val_X, te_X])
    y_all = np.concatenate([tr_y, val_y, te_y])
    
    # Split 1: Pisahkan 20% untuk TEST
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )
    
    # Split 2: Dari sisa, pisahkan 25% untuk VAL (25% x 80% = 20% total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def standardize(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train-mean)/std, (X_val-mean)/std, (X_test-mean)/std


# 2. ELM CORE
def elm_train(X, y, n_hidden=2000, lam=1e-3, random_state=42):
    """
    ELM dengan n_hidden besar (2000) untuk menangkap pola kompleks.
    """
    n_samples, n_features = X.shape
    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    
    rng = np.random.default_rng(random_state)
    W = rng.normal(0, 1, size=(n_hidden, n_features))
    b = rng.normal(0, 1, size=(n_hidden,))
    
    H = 1.0 / (1.0 + np.exp(-(X @ W.T + b)))
    
    Y = np.zeros((n_samples, n_classes))
    Y[np.arange(n_samples), y_idx] = 1.0
    
    HtH = H.T @ H
    I = np.eye(HtH.shape[0])
    beta = np.linalg.solve(HtH + lam * I, H.T @ Y)
    
    return {"W": W, "b": b, "beta": beta, "classes": classes}

def elm_predict(model, X):
    H = 1.0 / (1.0 + np.exp(-(X @ model["W"].T + model["b"])))
    scores = H @ model["beta"]
    return model["classes"][np.argmax(scores, axis=1)]


# 3. ACO FEATURE SELECTION

def aco_select(X_tr, y_tr, X_v, y_v, max_feat=160, n_ants=10, n_iter=20):
    n_feat = X_tr.shape[1]
    tau = np.ones(n_feat)
    best_mask = np.ones(n_feat, dtype=bool)
    best_score = 0.0
    rng = np.random.default_rng(42)
    
    print(f"  [ACO Info] Start Selection. Max Feat Allowed: {max_feat}")
    
    for it in range(n_iter):
        ant_masks, ant_scores = [], []
        tau_max = tau.max() + 1e-9
        
        for ant in range(n_ants):
            
            probs = np.clip(tau/tau_max, 0.1, 0.9)
            mask = rng.random(n_feat) < probs
            
            # (Pruning)
            if max_feat and mask.sum() > max_feat:
                
                on_indices = np.where(mask)[0]
                off = rng.choice(on_indices, mask.sum()-max_feat, replace=False)
                mask[off] = False
            
            
            if mask.sum() < 5: 
                add = rng.choice(np.where(~mask)[0], 5 - mask.sum(), replace=False)
                mask[add] = True
                

            model = elm_train(X_tr[:, mask], y_tr, n_hidden=800) 
            acc = accuracy_score(y_v, elm_predict(model, X_v[:, mask]))
            
            ant_masks.append(mask)
            ant_scores.append(acc)
            
            if acc > best_score:
                best_score = acc
                best_mask = mask.copy()
        
        # Update Pheromone
        tau *= 0.85 
        
        
        order = np.argsort(ant_scores)[::-1]
        for rank, idx in enumerate(order):
            if ant_scores[idx] <= 0: continue
            
            tau[ant_masks[idx]] += ant_scores[idx] / (rank + 1.0)
            
        print(f"    Iter {it+1}/{n_iter}: Best Val Acc={best_score:.4f}, Feats Selected={best_mask.sum()}")
        
    return best_mask


# 4. PIPELINE PENUH

def run_full_experiment(feature_type, split_mode, use_aco):
    mode_str = f"{feature_type.upper()} | {split_mode} SPLIT | ACO={use_aco}"
    print(f"\n[{mode_str}] Running...")
    
    # 1. Load Data
    if split_mode == "FIXED":
        data = load_data_fixed(feature_type)
    else:
        data = load_data_random(feature_type)
        
    if data is None: 
        print(f"  [ERR] Data {feature_type} not found. Pastikan file .npz ada.")
        return 0.0

    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = data
    X_tr, X_v, X_te = standardize(X_tr, X_v, X_te)
    
    # 2. ACO Feature Selection
    mask = np.ones(X_tr.shape[1], dtype=bool)
    if use_aco:
        print(f"  Running ACO selection...")

        mask = aco_select(X_tr, y_tr, X_v, y_v, max_feat=160, n_ants=10, n_iter=20)
        print(f"  [ACO Final] Features selected: {mask.sum()} / {X_tr.shape[1]}")
        
    # 3. Final Training & Testing
    X_final = np.vstack([X_tr[:, mask], X_v[:, mask]])
    y_final = np.concatenate([y_tr, y_v])
    
    print("  Training Final ELM Model...")
    model = elm_train(X_final, y_final, n_hidden=2000)
    y_pred = elm_predict(model, X_te[:, mask])
    acc = accuracy_score(y_te, y_pred)
    
    print(f"  >>> Accuracy: {acc:.4f}")
    
    # (Random Split + ACO) dan (Fixed Split + ACO)
    if use_aco:
         print("\nDetailed Classification Report:")
         print(classification_report(y_te, y_pred, zero_division=0))
    
    return acc


# MAIN EXECUTION

if __name__ == "__main__":
    results = []
    
    # Urutan Eksekusi:
    # 1. Shearlet (Jagoan Utama)
    # 2. Wavelet (Pembanding)
    
    for feat in ["shearlet", "wavelet"]:
        for split in ["FIXED", "RANDOM"]:
            for aco in [False, True]:
                acc = run_full_experiment(feat, split, aco)
                results.append({
                    "Feature": feat.upper(),
                    "Split": split,
                    "ACO": "Yes" if aco else "No",
                    "Accuracy": acc
                })

    print("\n\n" + "="*65)
    print(f"{'FEATURE':<10} | {'SPLIT TYPE':<10} | {'ACO':<5} | {'ACCURACY':<10}")
    print("="*65)
    for r in results:
        print(f"{r['Feature']:<10} | {r['Split']:<10} | {r['ACO']:<5} | {r['Accuracy']:.4%}")
    print("="*65) 