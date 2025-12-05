import numpy as np

def load_split_counts(txt_path, num_classes=25):
    counts = np.zeros(num_classes, dtype=int)
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[1])
            counts[label] += 1
    return counts

train_txt = "train-calibrated-shuffled.txt"
val_txt   = "val-calibrated-shuffled.txt"
test_txt  = "test-calibrated-shuffled.txt"


train_counts = load_split_counts(train_txt)
val_counts   = load_split_counts(val_txt)
test_counts  = load_split_counts(test_txt)

print("=== DISTRIBUSI DATA PER KELAS ===\n")

print("Kelas | Train | Val  | Test")
print("-------------------------------")

for cls in range(25):
    print(f"{cls:5d} | {train_counts[cls]:5d} | {val_counts[cls]:5d} | {test_counts[cls]:5d}")

print("\nTOTAL TRAIN:", train_counts.sum())
print("TOTAL VAL:  ", val_counts.sum())
print("TOTAL TEST: ", test_counts.sum())
