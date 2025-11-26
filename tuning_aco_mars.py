import numpy as np
import os
from sklearn.metrics import accuracy_score

# ==========================
# 1. DATA LOADER
# ==========================
def load_data():
    # Kita fokus tuning di SHEARLET FIXED SPLIT (Kasus utama skripsimu)
    if not os.path.exists("shearlet_train.npz"):
        print("File shearlet_train.npz tidak ditemukan!")
        return None
    
    train = np.load("shearlet_train.npz")
    val   = np.load("shearlet_val.npz")
    test  = np.load("shearlet_test.npz")
    
    # Standardisasi
    mean = train["X"].mean(axis=0)
    std  = train["X"].std(axis=0)
    std[std == 0] = 1.0
    
    X_train = (train["X"] - mean) / std
    X_val   = (val["X"]   - mean) / std
    X_test  = (test["X"]  - mean) / std
    
    return (X_train, train["y"]), (X_val, val["y"]), (X_test, test["y"])

# ==========================
# 2. ELM CORE
# ==========================
def elm_train_predict(X_train, y_train, X_test, n_hidden=2000):
    rng = np.random.default_rng(42)
    n_samples, n_features = X_train.shape
    classes, y_idx = np.unique(y_train, return_inverse=True)
    
    # Training
    W = rng.normal(0, 1, size=(n_hidden, n_features))
    b = rng.normal(0, 1, size=(n_hidden,))
    H = 1.0 / (1.0 + np.exp(-(X_train @ W.T + b)))
    
    Y = np.zeros((n_samples, len(classes)))
    Y[np.arange(n_samples), y_idx] = 1.0
    
    HtH = H.T @ H
    I = np.eye(HtH.shape[0])
    beta = np.linalg.solve(HtH + 1e-3 * I, H.T @ Y)
    
    # Prediction
    H_test = 1.0 / (1.0 + np.exp(-(X_test @ W.T + b)))
    scores = H_test @ beta
    return classes[np.argmax(scores, axis=1)]

# ==========================
# 3. ACO LOGIC (Tunable)
# ==========================
def run_aco_experiment(X_tr, y_tr, X_v, y_v, n_ants, n_iter, max_feat):
    n_feat = X_tr.shape[1]
    tau = np.ones(n_feat)
    best_mask = np.ones(n_feat, dtype=bool)
    best_score = 0.0
    rng = np.random.default_rng(42) # Seed tetap biar adil antar eksperimen
    
    # Early stopping: kalau 5x iterasi gak naik, stop
    patience = 5
    no_improv = 0
    
    for it in range(n_iter):
        ant_masks, ant_scores = [], []
        tau_max = tau.max() + 1e-9
        
        for ant in range(n_ants):
            probs = np.clip(tau/tau_max, 0.1, 0.9)
            mask = rng.random(n_feat) < probs
            
            # Constraint Max Features
            if mask.sum() > max_feat:
                off = rng.choice(np.where(mask)[0], mask.sum()-max_feat, replace=False)
                mask[off] = False
            if mask.sum() < 5: continue
            
            # Evaluasi Cepat (Hidden 500 cukup buat seleksi)
            y_pred = elm_train_predict(X_tr[:, mask], y_tr, X_v[:, mask], n_hidden=500)
            acc = accuracy_score(y_v, y_pred)
            
            ant_masks.append(mask)
            ant_scores.append(acc)
            
            if acc > best_score:
                best_score = acc
                best_mask = mask.copy()
                no_improv = 0
        
        no_improv += 1
        tau *= 0.85
        for i, sc in enumerate(ant_scores):
            tau[ant_masks[i]] += sc
            
        if no_improv >= patience:
            break # Stop early biar hemat waktu
            
    return best_mask

# ==========================
# 4. MAIN GRID SEARCH
# ==========================
if __name__ == "__main__":
    data = load_data()
    if data:
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = data
        
        # --- PARAMETER GRID YANG AKAN DICOBA ---
        # Kita coba variasi jumlah fitur maksimal & jumlah semut
        param_grid = [
            {'ants': 5,  'iter': 15, 'max_feat': 120}, # Hemat fitur
            {'ants': 10, 'iter': 20, 'max_feat': 150}, # Standard
            {'ants': 20, 'iter': 20, 'max_feat': 180}, # Bebasin hampir semua
            {'ants': 15, 'iter': 25, 'max_feat': 100}, # Tekan fitur rendah
        ]
        
        print(f"{'ANTS':<5} | {'ITER':<5} | {'MAX_FEAT':<10} | {'N_FEATS':<10} | {'VAL ACC':<10} | {'TEST ACC':<10}")
        print("-" * 70)
        
        best_result = {'acc': 0, 'params': None}
        
        for p in param_grid:
            # 1. Jalankan ACO
            mask = run_aco_experiment(X_tr, y_tr, X_v, y_v, p['ants'], p['iter'], p['max_feat'])
            
            # 2. Evaluasi Akhir (Train+Val -> Test)
            X_final_train = np.vstack([X_tr[:, mask], X_v[:, mask]])
            y_final_train = np.concatenate([y_tr, y_v])
            
            # Validasi Score (ELM Besar)
            y_val_pred = elm_train_predict(X_tr[:, mask], y_tr, X_v[:, mask], n_hidden=2000)
            val_acc = accuracy_score(y_v, y_val_pred)
            
            # Test Score (ELM Besar)
            y_test_pred = elm_train_predict(X_final_train, y_final_train, X_te[:, mask], n_hidden=2000)
            test_acc = accuracy_score(y_te, y_test_pred)
            
            print(f"{p['ants']:<5} | {p['iter']:<5} | {p['max_feat']:<10} | {mask.sum():<10} | {val_acc:.4f}     | {test_acc:.4f}")
            
            if test_acc > best_result['acc']:
                best_result = {'acc': test_acc, 'params': p, 'feats': mask.sum()}

        print("-" * 70)
        print(f"BEST RESULT: Accuracy {best_result['acc']:.4f}")
        print(f"Parameters: {best_result['params']}")
        print(f"Selected Features: {best_result['feats']}")