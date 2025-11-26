import numpy as np
import os

# ==========================
# 1. LOAD DATA SHEARLET
# ==========================

def load_shearlet_sets(root="."):
    train = np.load(os.path.join(root, "shearlet_train.npz"))
    val   = np.load(os.path.join(root, "shearlet_val.npz"))
    test  = np.load(os.path.join(root, "shearlet_test.npz"))

    X_train, y_train = train["X"], train["y"]
    X_val,   y_val   = val["X"],   val["y"]
    X_test,  y_test  = test["X"],  test["y"]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ==========================
# 2. STANDARDISASI FITUR
# ==========================

def standardize_train_val_test(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train_std = (X_train - mean) / std
    X_val_std   = (X_val   - mean) / std
    X_test_std  = (X_test  - mean) / std

    return X_train_std, X_val_std, X_test_std, mean, std


# ==========================
# 3. IMPLEMENTASI ELM
# ==========================

def elm_train(X, y, n_hidden=200, activation="sigmoid", lam=1e-3, random_state=None):
    """
    ELM multikelas sederhana:
      - 1 hidden layer, bobot acak
      - output layer via ridge regression
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape
    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    # bobot input & bias
    W = rng.normal(0, 1, size=(n_hidden, n_features))
    b = rng.normal(0, 1, size=(n_hidden,))

    # hidden layer
    H = X @ W.T + b
    if activation == "sigmoid":
        H = 1.0 / (1.0 + np.exp(-H))
    elif activation == "relu":
        H = np.maximum(0.0, H)
    else:
        raise ValueError("Unknown activation: %s" % activation)

    # one-hot target
    Y = np.zeros((n_samples, n_classes), dtype=np.float64)
    Y[np.arange(n_samples), y_idx] = 1.0

    # ridge regression: beta = (H^T H + lam I)^-1 H^T Y
    HtH = H.T @ H
    reg = lam * np.eye(HtH.shape[0])
    beta = np.linalg.solve(HtH + reg, H.T @ Y)

    model = {
        "W": W,
        "b": b,
        "beta": beta,
        "classes": classes,
        "activation": activation,
    }
    return model


def elm_predict(model, X):
    W = model["W"]
    b = model["b"]
    beta = model["beta"]
    classes = model["classes"]
    activation = model["activation"]

    H = X @ W.T + b
    if activation == "sigmoid":
        H = 1.0 / (1.0 + np.exp(-H))
    elif activation == "relu":
        H = np.maximum(0.0, H)
    scores = H @ beta
    idx = np.argmax(scores, axis=1)
    return classes[idx]


def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean()


# ==========================
# 4. EVALUASI SUBSET FITUR (UNTUK ACO)
# ==========================

def evaluate_subset(mask, X_train, y_train, X_val, y_val,
                    n_hidden=200, random_state=None):
    # kalau fitur terlalu sedikit, return 0
    if mask.sum() < 2:
        return 0.0

    X_tr = X_train[:, mask]
    X_v  = X_val[:, mask]

    model = elm_train(X_tr, y_train, n_hidden=n_hidden,
                      activation="sigmoid", lam=1e-3,
                      random_state=random_state)
    y_pred_val = elm_predict(model, X_v)
    acc = accuracy_score(y_val, y_pred_val)
    return acc


# ==========================
# 5. ACO UNTUK SELEKSI FITUR
# ==========================

def aco_feature_selection(
    X_train, y_train, X_val, y_val,
    n_ants=10,
    n_iterations=15,
    min_features=10,
    max_features=None,
    n_hidden=200,
    rho=0.2,
    random_state=0
):
    rng = np.random.default_rng(random_state)
    D = X_train.shape[1]  # jumlah fitur, harusnya 183

    # inisialisasi pheromone
    tau = np.ones(D, dtype=np.float64)

    best_mask = np.ones(D, dtype=bool)
    best_score = 0.0

    for it in range(n_iterations):
        ant_masks = []
        ant_scores = []

        tau_max = tau.max() + 1e-8

        for a in range(n_ants):
            # probabilitas pilih fitur dari pheromone
            probs = tau / tau_max
            probs = np.clip(probs, 0.05, 0.9)

            # sampling subset fitur
            mask = rng.random(D) < probs

            # pastikan min_features
            if mask.sum() < min_features:
                idx_add = rng.choice(
                    np.where(~mask)[0],
                    size=min_features - mask.sum(),
                    replace=False
                )
                mask[idx_add] = True

            # batasi max_features kalau di-set
            if max_features is not None and mask.sum() > max_features:
                idx_on = np.where(mask)[0]
                idx_off = rng.choice(
                    idx_on,
                    size=mask.sum() - max_features,
                    replace=False
                )
                mask[idx_off] = False

            # evaluasi subset pakai ELM di validation
            acc = evaluate_subset(
                mask, X_train, y_train, X_val, y_val,
                n_hidden=n_hidden,
                random_state=rng.integers(1e9)
            )

            ant_masks.append(mask)
            ant_scores.append(acc)

            if acc > best_score:
                best_score = acc
                best_mask = mask.copy()

        # pheromone evaporation
        tau *= (1.0 - rho)

        # deposit pheromone dari semut terbaik (rank-based)
        order = np.argsort(ant_scores)[::-1]
        for rank, idx_ant in enumerate(order):
            score = ant_scores[idx_ant]
            if score <= 0:
                continue
            mask = ant_masks[idx_ant]
            delta = score / (rank + 1.0)
            tau[mask] += delta

        print(f"Iter {it+1}/{n_iterations} - best val acc so far: {best_score:.4f}, "
              f"selected features: {best_mask.sum()}")

    return best_mask, best_score, tau


# ==========================
# 6. BASELINE ELM TANPA ACO
# ==========================

def baseline_elm_without_aco(X_train, y_train, X_val, y_val, X_test, y_test):
    # Standardisasi sama seperti biasa
    X_train_std, X_val_std, X_test_std, mean, std = standardize_train_val_test(
        X_train, X_val, X_test
    )

    # Gabungkan train + val untuk final training
    X_trval = np.vstack([X_train_std, X_val_std])
    y_trval = np.concatenate([y_train, y_val])

    # Train ELM (tanpa ACO, semua fitur dipakai)
    model = elm_train(
        X_trval, y_trval,
        n_hidden=200,
        activation="sigmoid",
        lam=1e-3,
        random_state=123
    )

    # Test
    y_pred_test = elm_predict(model, X_test_std)
    acc_test = accuracy_score(y_test, y_pred_test)

    print("\n=== BASELINE ELM (NO ACO) ===")
    print("Test accuracy:", acc_test)

    return acc_test


# ==========================
# 7. MAIN PIPELINE: BASELINE -> ACO -> FINAL ELM
# ==========================

def main():
    # 1) load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_shearlet_sets(".")
    print("Loaded:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)
    print("  X_test: ", X_test.shape,  "y_test: ", y_test.shape)

    # 2) baseline tanpa ACO
    baseline_elm_without_aco(X_train, y_train, X_val, y_val, X_test, y_test)

    # 3) standardisasi untuk ACO
    X_train_std, X_val_std, X_test_std, mean, std = standardize_train_val_test(
        X_train, X_val, X_test
    )

    # 4) ACO feature selection (train+val)
    best_mask, best_val_score, tau = aco_feature_selection(
        X_train_std, y_train,
        X_val_std,   y_val,
        n_ants=10,
        n_iterations=15,
        min_features=10,
        max_features=80,    # dibatasi maksimal 80 fitur
        n_hidden=200,
        rho=0.2,
        random_state=42,
    )

    print("\n=== ACO RESULT ===")
    print("Best val accuracy:", best_val_score)
    print("Selected features:", best_mask.sum(), "dari", X_train.shape[1])

    # 5) Train final ELM di TRAIN+VAL dengan fitur terpilih
    X_trval = np.vstack([X_train_std, X_val_std])[:, best_mask]
    y_trval = np.concatenate([y_train, y_val])

    print("Training final ELM on train+val (with selected features)...")
    final_model = elm_train(
        X_trval, y_trval,
        n_hidden=200,
        activation="sigmoid",
        lam=1e-3,
        random_state=123
    )

    # 6) Evaluasi di TEST
    X_test_sel = X_test_std[:, best_mask]
    y_pred_test = elm_predict(final_model, X_test_sel)
    acc_test = accuracy_score(y_test, y_pred_test)

    print("\n=== FINAL TEST PERFORMANCE ===")
    print("Test accuracy (ACO+ELM with shearlet features):", acc_test)


if __name__ == "__main__":
    main()
