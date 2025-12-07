import numpy as np
import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score

# ==========================
# 1. DATA LOADING
# ==========================
def load_shearlet_data():
    if not os.path.exists("shearlet_train.npz"):
        print("[ERROR] File shearlet_train.npz tidak ditemukan!")
        return None
    
    print("Loading Shearlet Data...")
    train = np.load("shearlet_train.npz")
    val   = np.load("shearlet_val.npz")
    test  = np.load("shearlet_test.npz")
    
    X_tr, y_tr = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]
    X_te, y_te = test["X"], test["y"]
    
    # Standardisasi
    mean = X_tr.mean(axis=0)
    std  = X_tr.std(axis=0)
    std[std == 0] = 1.0
    
    X_tr  = (X_tr - mean) / std
    X_val = (X_val - mean) / std
    X_te  = (X_te - mean) / std
    
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)

# ==========================
# 2. ELM MODEL
# ==========================
def elm_train(X, y, n_hidden=2000, lam=1e-3):
    n_samples, n_features = X.shape
    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    
    rng = np.random.default_rng(42)
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

# ==========================
# 3. SMART ACO ENGINE (SIGMOID + SMART PRUNING)
# ==========================
def run_aco_search(X_tr, y_tr, X_v, y_v, params):
    n_ants = params['ants']
    n_iter = params['iter']
    max_feat = params['max_feat']
    
    n_feat = X_tr.shape[1]
    
    # Inisialisasi Tau = 1.0
    tau = np.ones(n_feat)
    K_factor = 1.0 
    
    best_mask = np.ones(n_feat, dtype=bool)
    best_val_acc = 0.0
    
    rng = np.random.default_rng(42)
    patience = 8
    no_improv = 0
    
    for it in range(n_iter):
        ant_masks, ant_scores = [], []
        
        # Dinamika Pheromone: Sigmoid
        probs = tau / (tau + K_factor)
        probs = np.clip(probs, 0.05, 0.95)
        
        for ant in range(n_ants):
            # 1. Seleksi Fitur
            mask = rng.random(n_feat) < probs
            
            # 2. SMART PRUNING (Jika kelebihan muatan)
            current_k = mask.sum()
            if max_feat and current_k > max_feat:
                on_idx = np.where(mask)[0]
                on_tau = tau[on_idx]
                
                # Sort berdasarkan pheromone terlemah (+ sedikit noise random)
                sort_metric = on_tau + rng.uniform(0, 0.01, size=len(on_tau))
                sorted_idx = np.argsort(sort_metric)
                
                # Buang fitur terlemah
                n_remove = current_k - max_feat
                remove_local_idx = sorted_idx[:n_remove]
                mask[on_idx[remove_local_idx]] = False
            
            # Smart Addition (Jika terlalu sedikit < 5)
            if mask.sum() < 5:
                off_idx = np.where(~mask)[0]
                if len(off_idx) > 0:
                    off_tau = tau[off_idx]
                    prob_revive = off_tau / (off_tau.sum() + 1e-9)
                    add_idx = rng.choice(off_idx, size=min(5, len(off_idx)), p=prob_revive, replace=False)
                    mask[add_idx] = True

            # 3. Evaluasi (Low Hidden Nodes untuk kecepatan)
            if mask.sum() == 0: continue
            
            model = elm_train(X_tr[:, mask], y_tr, n_hidden=800)
            acc = accuracy_score(y_v, elm_predict(model, X_v[:, mask]))
            
            ant_masks.append(mask)
            ant_scores.append(acc)
            
            if acc > best_val_acc:
                best_val_acc = acc
                best_mask = mask.copy()
                no_improv = 0
        
        no_improv += 1
        
        # 4. Update Pheromone
        tau *= 0.85 # Evaporasi
        
        order = np.argsort(ant_scores)[::-1]
        n_elite = max(1, n_ants // 2)
        
        for i in range(n_elite):
            idx = order[i]
            score = ant_scores[idx]
            if score <= 0: continue
            tau[ant_masks[idx]] += score 
            
        if no_improv >= patience: break
            
    return best_mask

# ==========================
# 4. MAIN GRID SEARCH (EXTREME MODE)
# ==========================
if __name__ == "__main__":
    data = load_shearlet_data()
    
    if data:
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = data
        
        # --- GRID SEARCH "EXTREME" ---
        # Mencari batas minimal fitur dengan swarm besar
        
        # Semut: Ditambah untuk daya jelajah lebih tinggi di ruang sempit
        ants_options = [30, 35, 40]
        
        # Iterasi: Cukup pendek karena konvergensi cepat
        iter_options = [10, 15]
        
        # Fitur: Sangat ketat (40-85 fitur)
        feat_options = [40, 50, 65, 85]
        
        results = []
        total_runs = len(ants_options) * len(iter_options) * len(feat_options)
        counter = 1
        
        print("\n" + "="*90)
        print(f"STARTING EXTREME TUNING ({total_runs} Combinations)")
        print(f"Ants: {ants_options}")
        print(f"Iter: {iter_options}")
        print(f"Feat: {feat_options}")
        print("="*90)
        
        start_time = time.time()
        
        for ants in ants_options:
            for it in iter_options:
                for mf in feat_options:
                    
                    params = {'ants': ants, 'iter': it, 'max_feat': mf}
                    print(f"[{counter:02d}/{total_runs}] Ants={ants}, Iter={it}, Limit={mf} ... ", end="")
                    
                    st_run = time.time()
                    mask = run_aco_search(X_tr, y_tr, X_v, y_v, params)
                    sel_feats = mask.sum()
                    
                    # Validasi (High Hidden Nodes)
                    model_val = elm_train(X_tr[:, mask], y_tr, n_hidden=2000)
                    val_pred = elm_predict(model_val, X_v[:, mask])
                    val_acc = accuracy_score(y_v, val_pred)
                    
                    # Testing (Intip)
                    X_final_tr = np.vstack([X_tr[:, mask], X_v[:, mask]])
                    y_final_tr = np.concatenate([y_tr, y_v])
                    
                    model_test = elm_train(X_final_tr, y_final_tr, n_hidden=2000)
                    test_pred = elm_predict(model_test, X_te[:, mask])
                    test_acc = accuracy_score(y_te, test_pred)
                    
                    dur = time.time() - st_run
                    print(f"Done ({dur:4.1f}s) | Val: {val_acc:.4f} | Test: {test_acc:.4f} | Got: {sel_feats} feats")
                    
                    results.append({
                        'Ants': ants,
                        'Iter': it,
                        'Max_Limit': mf,
                        'Selected_Feats': sel_feats,
                        'Val_Acc': val_acc,
                        'Test_Acc': test_acc,
                        'Time': dur
                    })
                    counter += 1

        print("\n" + "="*90)
        print("EXTREME TUNING RESULTS (Sorted by Val Accuracy)")
        print("="*90)
        
        df = pd.DataFrame(results)
        df = df.sort_values(by=['Val_Acc', 'Test_Acc', 'Selected_Feats'], ascending=[False, False, True])
        
        print(df.to_string(index=False))
        df.to_csv("aco_extreme_results.csv", index=False)
        
        print(f"\nTotal Waktu: {(time.time() - start_time)/60:.2f} menit")