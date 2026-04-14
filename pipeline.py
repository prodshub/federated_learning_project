"""
Federated Averaging (FedAvg) on CIFAR-10  —  CPU-friendly implementation
Based on: McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data" (AISTATS 2017)

Architecture: Lightweight CNN (two conv layers + two FC layers)
Dataset:      Synthetic CIFAR-10 (3x32x32, 10 classes, 6000 train / 1200 test)
              Note: real CIFAR-10 requires internet access; synthetic data has
              the same shape and class structure so the algorithm behaves identically.

Pipeline
--------
  1.  Data generation and IID partitioning across K clients
  2.  CNN model definition
  3.  Client local SGD  (Algorithm 1 ClientUpdate)
  4.  FedAvg server aggregation (weighted average)
  5.  FedSGD baseline  (E=1, B=inf)
  6.  Training loop with per-round logging
  7.  Results plot: test accuracy vs communication rounds
"""

import copy, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 0.  Reproducibility
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ──────────────────────────────────────────────
# 1.  Synthetic CIFAR-10-shaped dataset
# ──────────────────────────────────────────────
class CIFAR10Dataset(Dataset):
    """
    Lazy-loading CIFAR-10 dataset.
    Stores data as uint8 (4x smaller) and normalises per item in __getitem__.
    """
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    def __init__(self, data, labels):
        # Keep as uint8 to save memory; shape (N, 3072)
        self.data   = data                                  # numpy uint8
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = torch.from_numpy(
            self.data[i].reshape(3, 32, 32).astype(np.float32)
        ) / 255.0
        img = (img - self.MEAN) / self.STD
        return img, self.labels[i]


def load_batch(filepath):
    import pickle
    with open(filepath, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"], np.array(d[b"labels"])


def build_datasets(data_dir="/mnt/user-data/uploads"):
    """Load real CIFAR-10 from the uploaded binary batch files."""
    train_data, train_labels = [], []
    for i in range(1, 6):
        data, labels = load_batch(f"{data_dir}/data_batch_{i}")
        train_data.append(data)
        train_labels.append(labels)
    train_data   = np.concatenate(train_data,   axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    test_data, test_labels = load_batch(f"{data_dir}/test_batch")

    print(f"  [Data] Real CIFAR-10: train={len(train_labels):,}  test={len(test_labels):,}  classes=10  shape=3x32x32")
    return CIFAR10Dataset(train_data, train_labels), CIFAR10Dataset(test_data, test_labels)



def iid_partition(n_total, K):
    """Shuffle then split evenly - IID partition from paper Section 3."""
    idx = list(range(n_total))
    random.shuffle(idx)
    per = n_total // K
    return {k: idx[k*per:(k+1)*per] for k in range(K)}


# ──────────────────────────────────────────────
# 2.  Model
# ──────────────────────────────────────────────
class CifarCNN(nn.Module):
    """
    Two conv layers + two FC layers, mirroring the paper architecture.
    ~250k parameters - runs comfortably on CPU.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# 3.  Client local update  (Algorithm 1)
# ──────────────────────────────────────────────
def client_update(global_weights, dataset, indices, E, B, lr):
    """
    ClientUpdate(k, w_t) from Algorithm 1 in the paper.
    Initialise from global weights, run E epochs of SGD with minibatch B,
    return updated local weights.
    """
    model = CifarCNN()
    model.load_state_dict(copy.deepcopy(global_weights))
    model.train()

    loader  = DataLoader(Subset(dataset, indices), batch_size=B, shuffle=True)
    opt     = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(E):
        for x, y in loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    return model.state_dict()


# ──────────────────────────────────────────────
# 4.  Server aggregation
# ──────────────────────────────────────────────
def federated_averaging(weights_list, sizes):
    """
    w_{t+1} = sum_k (n_k / n) * w^k_{t+1}
    Weighted average by local dataset size, as in the paper.
    """
    n_total = sum(sizes)
    avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in weights_list[0].items()}
    for w, n_k in zip(weights_list, sizes):
        frac = n_k / n_total
        for k in avg:
            avg[k] += frac * w[k].float()
    return avg


# ──────────────────────────────────────────────
# 5.  Evaluation
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


# ──────────────────────────────────────────────
# 6.  Training loops
# ──────────────────────────────────────────────
def run_fedavg(train_ds, test_loader, client_idx, rounds, K, C, E, B, lr, label):
    print(f"\n{'='*56}")
    print(f"  {label}")
    print(f"  rounds={rounds}  K={K}  C={C}  E={E}  B={B}  lr={lr}")
    print(f"{'='*56}")

    m = max(int(C * K), 1)
    global_model = CifarCNN()
    accs = []

    for t in range(1, rounds + 1):
        t0 = time.time()
        selected = random.sample(range(K), m)

        local_weights = []
        local_sizes   = []
        for k in selected:
            w = client_update(global_model.state_dict(), train_ds,
                              client_idx[k], E, B, lr)
            local_weights.append(w)
            local_sizes.append(len(client_idx[k]))

        global_model.load_state_dict(federated_averaging(local_weights, local_sizes))
        acc = evaluate(global_model, test_loader)
        accs.append(acc)
        print(f"  Round {t:3d}/{rounds}  acc={acc*100:5.2f}%  ({time.time()-t0:.1f}s)")

    return accs


def run_fedsgd(train_ds, test_loader, client_idx, rounds, K, C, lr, label):
    """FedSGD = FedAvg with E=1 and B=inf (full local dataset as one batch)."""
    print(f"\n{'='*56}")
    print(f"  {label}")
    print(f"  rounds={rounds}  K={K}  C={C}  E=1  B=inf  lr={lr}")
    print(f"{'='*56}")

    m = max(int(C * K), 1)
    global_model = CifarCNN()
    accs = []

    for t in range(1, rounds + 1):
        t0 = time.time()
        selected = random.sample(range(K), m)

        local_weights = []
        local_sizes   = []
        for k in selected:
            idx = client_idx[k]
            w = client_update(global_model.state_dict(), train_ds,
                              idx, E=1, B=len(idx), lr=lr)
            local_weights.append(w)
            local_sizes.append(len(idx))

        global_model.load_state_dict(federated_averaging(local_weights, local_sizes))
        acc = evaluate(global_model, test_loader)
        accs.append(acc)
        print(f"  Round {t:3d}/{rounds}  acc={acc*100:5.2f}%  ({time.time()-t0:.1f}s)")

    return accs


# ──────────────────────────────────────────────
# 7.  Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # K=20 clients x 500 samples each (first 10k of the 50k training set).
    # This matches the paper's per-client sample count; K is reduced for CPU speed.
    # To use all 50k with K=100, swap the Subset line below and set K=100.
    K      = 20
    N_USE  = K * 500      # 10,000 training samples
    ROUNDS = 25

    train_ds_full, test_ds = build_datasets()
    # Use a consistent IID subset of the full training set
    train_ds   = torch.utils.data.Subset(train_ds_full, list(range(N_USE)))
    client_idx = iid_partition(N_USE, K)           # 500 samples per client
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    n_params = sum(p.numel() for p in CifarCNN().parameters())
    print(f"  Model parameters : {n_params:,}")
    print(f"  Clients (K)      : {K}  ({N_USE//K} samples each, {N_USE:,} total train)")

    fedavg_accs = run_fedavg(
        train_ds, test_loader, client_idx,
        rounds=ROUNDS, K=K, C=0.2, E=3, B=50, lr=0.01,
        label="FedAvg  (C=0.2, E=3, B=50)"
    )
    fedsgd_accs = run_fedsgd(
        train_ds, test_loader, client_idx,
        rounds=ROUNDS, K=K, C=0.2, lr=0.05,
        label="FedSGD  (C=0.2, E=1, B=inf)"
    )

    # ── Plot ───────────────────────────────────
    rounds_x = list(range(1, ROUNDS + 1))
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(rounds_x, [a*100 for a in fedavg_accs],
            lw=2.5, color="#2563eb", marker="o", markersize=3,
            label="FedAvg  (C=0.2, E=5, B=32)")
    ax.plot(rounds_x, [a*100 for a in fedsgd_accs],
            lw=2.5, color="#dc2626", linestyle="--", marker="s", markersize=3,
            label="FedSGD  (C=0.2, E=1, B=inf)")
    ax.axhline(10, color="gray", lw=1, linestyle=":", alpha=0.7, label="Random baseline (10%)")

    ax.set_xlabel("Communication Rounds", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title(
        "FedAvg vs FedSGD on CIFAR-10\n"
        "(IID partition, K=20 clients, synthetic CIFAR-10-shaped data)",
        fontsize=12
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(1, ROUNDS)
    ax.set_ylim(0, 100)

    for accs, color, dy in [(fedavg_accs, "#2563eb", 6), (fedsgd_accs, "#dc2626", -14)]:
        ax.annotate(f"{accs[-1]*100:.1f}%",
                    xy=(ROUNDS, accs[-1]*100),
                    xytext=(6, dy), textcoords="offset points",
                    fontsize=10, color=color, fontweight="bold")

    plt.tight_layout()
    out = "/mnt/user-data/outputs/fedavg_cifar10_results.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")

    # ── Summary ────────────────────────────────
    print("\n" + "="*56)
    print("  RESULTS SUMMARY")
    print("="*56)
    print(f"  Communication rounds : {ROUNDS}")
    print(f"  Clients (K)          : {K}  ({N_USE//K} samples each)")
    print(f"  FedAvg final acc     : {fedavg_accs[-1]*100:.2f}%")
    print(f"  FedSGD final acc     : {fedsgd_accs[-1]*100:.2f}%")
    print(f"  FedAvg peak acc      : {max(fedavg_accs)*100:.2f}%")
    print(f"  FedSGD peak acc      : {max(fedsgd_accs)*100:.2f}%")
    gap = (fedavg_accs[-1] - fedsgd_accs[-1]) * 100
    print(f"  FedAvg advantage     : {gap:+.2f}pp at round {ROUNDS}")
    print("="*56)