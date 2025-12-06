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
# 3. ACO ENGINE (SMART LOGIC)
# ==========================
def run_aco_search(X_tr, y_tr, X_v, y_v, params):
    n_ants = params['ants']
    n_iter = params['iter']
    max_feat = params['max_feat']
    
    n_feat = X_tr.shape[1]
    
    # Inisialisasi Tau = 1.0
    tau = np.ones(n_feat)
    
    # K factor untuk probabilitas sigmoid: P = tau / (tau + K)
    # K=1.0 artinya jika tau=1, prob=0.5 (Start Netral)
    K_factor = 1.0 
    
    best_mask = np.ones(n_feat, dtype=bool)
    best_val_acc = 0.0
    
    rng = np.random.default_rng(42)
    patience = 8
    no_improv = 0
    
    for it in range(n_iter):
        ant_masks, ant_scores = [], []
        
        # Dinamika Pheromone: Hitung probabilitas tiap fitur
        # Rumus Sigmoid: Probabilitas bergerak smooth dari 0 ke 1
        probs = tau / (tau + K_factor)
        
        # Clip sedikit biar tidak absolut 0 atau 1 (masih ada peluang eksplorasi)
        probs = np.clip(probs, 0.05, 0.95)
        
        for ant in range(n_ants):
            # 1. Seleksi Fitur Probabilistik
            mask = rng.random(n_feat) < probs
            
            # 2. SMART PRUNING (Jika kelebihan muatan)
            current_k = mask.sum()
            if current_k > max_feat:
                # Cari fitur yang aktif
                on_idx = np.where(mask)[0]
                
                # Lihat pheromone mereka. Jika masih awal (tau sama semua), tambahkan noise random biar adil
                # Jika sudah jalan, pheromone akan beda-beda.
                on_tau = tau[on_idx]
                
                # Tambah sedikit noise random saat sorting agar jika pheromone sama, yang dibuang random
                # (Penting untuk iterasi pertama biar gak deterministik buang fitur index awal)
                sort_metric = on_tau + rng.uniform(0, 0.01, size=len(on_tau))
                
                # Urutkan dari yang paling LEMAH pheromone-nya
                sorted_idx = np.argsort(sort_metric)
                
                # Ambil index fitur terlemah sebanyak kelebihannya
                n_remove = current_k - max_feat
                remove_local_idx = sorted_idx[:n_remove]
                
                # Matikan fitur tersebut
                remove_global_idx = on_idx[remove_local_idx]
                mask[remove_global_idx] = False
            
            # 3. SMART ADDITION (Jika terlalu sedikit < 5)
            if mask.sum() < 5:
                off_idx = np.where(~mask)[0]
                if len(off_idx) > 0:
                    # Pilih fitur mati yang pheromone-nya TINGGI untuk dihidupkan
                    off_tau = tau[off_idx]
                    prob_revive = off_tau / off_tau.sum()
                    add_idx = rng.choice(off_idx, size=min(5, len(off_idx)), p=prob_revive, replace=False)
                    mask[add_idx] = True

            # 4. Evaluasi
            if mask.sum() == 0: continue # Jaga-jaga
            
            model = elm_train(X_tr[:, mask], y_tr, n_hidden=800)
            acc = accuracy_score(y_v, elm_predict(model, X_v[:, mask]))
            
            ant_masks.append(mask)
            ant_scores.append(acc)
            
            if acc > best_val_acc:
                best_val_acc = acc
                best_mask = mask.copy()
                no_improv = 0
        
        no_improv += 1
        
        # 5. Pheromone Update
        tau *= 0.85 # Evaporasi
        
        order = np.argsort(ant_scores)[::-1]
        
        # Elite Strategy: Hanya separuh semut terbaik yang kasih feromon
        # Biar fitur jelek gak dapet reward dari semut yang "kebetulan" hoki
        n_elite = max(1, n_ants // 2)
        
        for i in range(n_elite):
            idx = order[i]
            score = ant_scores[idx]
            if score <= 0: continue
            
            # Reward: Semakin tinggi ranking, semakin besar
            reward = score 
            tau[ant_masks[idx]] += reward
            
        if no_improv >= patience: break
            
    return best_mask

# ==========================
# 4. MAIN GRID SEARCH (10-30 ANTS)
# ==========================
if __name__ == "__main__":
    data = load_shearlet_data()
    
    if data:
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = data
        
        # --- GRID SEARCH SYSTEMATIC ---
        ants_options = [10, 15, 20, 25, 30]
        iter_options = [10, 15, 20, 25, 30]
        
        # Kita masukkan 189 sebagai baseline
        feat_options = [85, 105, 125, 145, 165, 189]
        
        results = []
        total_runs = len(ants_options) * len(iter_options) * len(feat_options)
        counter = 1
        
        print("\n" + "="*90)
        print(f"STARTING SMART ACO GRID SEARCH ({total_runs} Combinations)")
        print("="*90)
        
        start_time = time.time()
        
        for ants in ants_options:
            for it in iter_options:
                for mf in feat_options:
                    
                    params = {'ants': ants, 'iter': it, 'max_feat': mf}
                    print(f"[{counter:03d}/{total_runs}] Ants={ants}, Iter={it}, Limit={mf} ... ", end="")
                    
                    st_run = time.time()
                    mask = run_aco_search(X_tr, y_tr, X_v, y_v, params)
                    sel_feats = mask.sum()
                    
                    # Validasi
                    model_val = elm_train(X_tr[:, mask], y_tr, n_hidden=2000)
                    val_pred = elm_predict(model_val, X_v[:, mask])
                    val_acc = accuracy_score(y_v, val_pred)
                    
                    # Testing
                    X_final_tr = np.vstack([X_tr[:, mask], X_v[:, mask]])
                    y_final_tr = np.concatenate([y_tr, y_v])
                    
                    model_test = elm_train(X_final_tr, y_final_tr, n_hidden=2000)
                    test_pred = elm_predict(model_test, X_te[:, mask])
                    test_acc = accuracy_score(y_te, test_pred)
                    
                    dur = time.time() - st_run
                    print(f"Done ({dur:4.1f}s) | Val: {val_acc:.4f} | Test: {test_acc:.4f} | Got: {sel_feats:3d} feats")
                    
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
        print("FINAL RESULTS (Sorted by Val Accuracy)")
        print("="*90)
        
        df = pd.DataFrame(results)
        df = df.sort_values(by=['Val_Acc', 'Test_Acc', 'Selected_Feats'], ascending=[False, False, True])
        
        print(df.to_string(index=False))
        df.to_csv("aco_smart_results.csv", index=False)
        
        print(f"\nTotal Waktu: {(time.time() - start_time)/60:.2f} menit")