"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           FEDERATED LEARNING SIMULATION FRAMEWORK                           ║
║  OOP-based | Multi-Dataset | Multi-Model | GPU-Accelerated | Rich Metrics   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture:
  FederatedClient  — owns local data, model replica, trains locally
  FederatedServer  — holds global model, broadcasts, aggregates
  FederatedSimulation — orchestrates the full FL loop
  DataDistributor  — splits datasets across clients (IID / Non-IID)
  SimConfig        — single config dataclass for all hyperparams

Supported:
  Datasets     : MNIST, CIFAR-10, FashionMNIST
  Models       : SimpleMLP, SimpleCNN, ResNet-lite
  Aggregation  : FedAvg, FedProx, FedMedian, FedAdam
  Distribution : IID, Non-IID Dirichlet, Pathological (shard-based)
"""

import os
import time
import copy
import random
import warnings
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# ANSI color helpers for terminal output
# ──────────────────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
MAGENTA= "\033[95m"
BLUE   = "\033[94m"
DIM    = "\033[2m"

def _c(color, text):  return f"{color}{text}{RESET}"
def header(msg):      print(f"\n{BOLD}{CYAN}{'═'*70}{RESET}\n{BOLD}{CYAN}  {msg}{RESET}\n{BOLD}{CYAN}{'═'*70}{RESET}")
def step(tag, msg):   print(f"  {BOLD}{YELLOW}[{tag}]{RESET} {msg}")
def info(msg):        print(f"  {DIM}{msg}{RESET}")
def success(msg):     print(f"  {BOLD}{GREEN}✔  {msg}{RESET}")
def warn(msg):        print(f"  {BOLD}{RED}⚠  {msg}{RESET}")
def divider():        print(f"  {DIM}{'─'*66}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimConfig:
    """All hyperparameters and knobs for a federated learning experiment."""

    # ── Dataset & Model ─────────────────────────────────────────────────────
    dataset: str = "mnist"           # 'mnist' | 'cifar10' | 'fashionmnist'
    model: str   = "cnn"             # 'mlp'   | 'cnn'     | 'resnet_lite'

    # ── Federation topology ──────────────────────────────────────────────────
    num_clients: int   = 10          # Total clients in the federation
    clients_per_round: float = 0.5   # Fraction selected each round (0 < x ≤ 1)
    num_rounds: int    = 15          # Global communication rounds
    local_epochs: int  = 3           # Local training epochs per round

    # ── Optimisation ────────────────────────────────────────────────────────
    learning_rate: float = 0.01
    batch_size: int      = 32
    weight_decay: float  = 1e-4

    # ── Aggregation ─────────────────────────────────────────────────────────
    aggregation: str     = "fedavg"  # 'fedavg' | 'fedprox' | 'fedmedian' | 'fedadam'
    fedprox_mu: float    = 0.01      # Proximal term (FedProx only)
    fedadam_lr: float    = 1e-3      # Server-side Adam LR (FedAdam only)
    fedadam_beta1: float = 0.9
    fedadam_beta2: float = 0.99

    # ── Data distribution ───────────────────────────────────────────────────
    distribution: str   = "iid"      # 'iid' | 'non_iid_dirichlet' | 'pathological'
    dirichlet_alpha: float = 0.5     # Concentration (lower = more heterogeneous)
    shards_per_client: int = 2       # For pathological: classes per client

    # ── System ──────────────────────────────────────────────────────────────
    use_gpu: bool = True
    seed: int     = 42
    data_dir: str = "./data"
    output_dir: str = "./fl_results"
    verbose_clients: bool = False    # Print per-client loss each epoch

    @property
    def device(self) -> torch.device:
        if self.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        if self.use_gpu and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def summary(self):
        header("SIMULATION CONFIGURATION")
        rows = [
            ("Dataset", self.dataset.upper()),
            ("Model", self.model.upper()),
            ("Clients (total / per round)", f"{self.num_clients} / {int(self.num_clients * self.clients_per_round)}"),
            ("Rounds × Local epochs", f"{self.num_rounds} × {self.local_epochs}"),
            ("Learning rate / Batch size", f"{self.learning_rate} / {self.batch_size}"),
            ("Aggregation", self.aggregation.upper()),
            ("Data distribution", self.distribution),
            ("Dirichlet α", self.dirichlet_alpha if "dirichlet" in self.distribution else "N/A"),
            ("Device", str(self.device).upper()),
            ("Random seed", self.seed),
        ]
        for k, v in rows:
            print(f"  {BOLD}{k:<36}{RESET}{CYAN}{v}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODELS
# ══════════════════════════════════════════════════════════════════════════════

class SimpleMLP(nn.Module):
    """Two-hidden-layer MLP. Fast baseline for MNIST / FashionMNIST."""

    def __init__(self, input_dim: int = 784, hidden: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """Lightweight CNN suitable for MNIST / CIFAR-10.
    Classifier is initialised eagerly so state_dicts are consistent."""

    _DEFAULT_SIZE = {1: 28, 3: 32}   # typical spatial size per channel count

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )
        # Compute flat size eagerly so classifier keys exist before any forward pass
        sz = self._DEFAULT_SIZE.get(in_channels, 28)
        with torch.no_grad():
            feat = self.features(torch.zeros(1, in_channels, sz, sz))
            flat = feat.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResNetLite(nn.Module):
    """Residual blocks — a step up for CIFAR-10."""

    class _ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
            )
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(x + self.block(x))

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.layer1 = self._ResBlock(64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            self._ResBlock(128),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            self._ResBlock(256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.layer3(self.layer2(self.layer1(self.stem(x)))))


def build_model(cfg: SimConfig) -> nn.Module:
    ds_info = {
        "mnist":       (1, 10, 28 * 28),
        "fashionmnist":(1, 10, 28 * 28),
        "cifar10":     (3, 10, 32 * 32 * 3),
    }
    in_ch, n_cls, flat = ds_info[cfg.dataset]

    if cfg.model == "mlp":
        return SimpleMLP(input_dim=flat, num_classes=n_cls)
    elif cfg.model == "cnn":
        return SimpleCNN(in_channels=in_ch, num_classes=n_cls)
    elif cfg.model == "resnet_lite":
        return ResNetLite(in_channels=in_ch, num_classes=n_cls)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA LOADING & DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

class DataDistributor:
    """Loads a dataset and splits indices across N clients."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.train_dataset, self.test_dataset = self._load()
        self.num_classes = len(self.train_dataset.classes)

    def _load(self):
        ds = self.cfg.dataset
        root = self.cfg.data_dir

        if ds == "mnist":
            T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            train = torchvision.datasets.MNIST(root, train=True,  download=True, transform=T)
            test  = torchvision.datasets.MNIST(root, train=False, download=True, transform=T)

        elif ds == "fashionmnist":
            T = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ])
            train = torchvision.datasets.FashionMNIST(root, train=True,  download=True, transform=T)
            test  = torchvision.datasets.FashionMNIST(root, train=False, download=True, transform=T)

        elif ds == "cifar10":
            T_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616)),
            ])
            T_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616)),
            ])
            train = torchvision.datasets.CIFAR10(root, train=True,  download=True, transform=T_train)
            test  = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=T_test)
        else:
            raise ValueError(f"Unknown dataset: {ds}")

        return train, test

    def split(self) -> List[List[int]]:
        """Returns a list of index lists, one per client."""
        method = self.cfg.distribution
        if method == "iid":
            return self._iid()
        elif method == "non_iid_dirichlet":
            return self._dirichlet()
        elif method == "pathological":
            return self._pathological()
        else:
            raise ValueError(f"Unknown distribution: {method}")

    def _iid(self) -> List[List[int]]:
        n = len(self.train_dataset)
        idx = list(range(n))
        random.shuffle(idx)
        size = n // self.cfg.num_clients
        return [idx[i * size:(i + 1) * size] for i in range(self.cfg.num_clients)]

    def _dirichlet(self) -> List[List[int]]:
        targets = np.array(self.train_dataset.targets)
        n_cls   = self.num_classes
        alpha   = self.cfg.dirichlet_alpha
        n       = self.cfg.num_clients

        client_idx = [[] for _ in range(n)]
        for c in range(n_cls):
            class_idx = np.where(targets == c)[0]
            np.random.shuffle(class_idx)
            proportions = np.random.dirichlet(np.repeat(alpha, n))
            proportions = (proportions * len(class_idx)).astype(int)
            # fix rounding
            proportions[-1] = len(class_idx) - proportions[:-1].sum()
            splits = np.split(class_idx, np.cumsum(proportions)[:-1])
            for cid, s in enumerate(splits):
                client_idx[cid].extend(s.tolist())
        return client_idx

    def _pathological(self) -> List[List[int]]:
        """Each client gets data from only `shards_per_client` classes."""
        targets = np.array(self.train_dataset.targets)
        n_cls   = self.num_classes
        n       = self.cfg.num_clients
        spc     = self.cfg.shards_per_client

        # Sort all samples by label
        sorted_idx = np.argsort(targets)
        shard_size = len(sorted_idx) // (n * spc)
        shards     = [sorted_idx[i * shard_size:(i + 1) * shard_size]
                      for i in range(n * spc)]
        random.shuffle(shards)

        client_idx = []
        for i in range(n):
            mine = shards[i * spc:(i + 1) * spc]
            client_idx.append(np.concatenate(mine).tolist())
        return client_idx

    def label_distribution(self, client_indices: List[List[int]]) -> np.ndarray:
        """Returns [num_clients x num_classes] count matrix."""
        targets = np.array(self.train_dataset.targets)
        dist    = np.zeros((len(client_indices), self.num_classes), dtype=int)
        for cid, idx in enumerate(client_indices):
            for label in targets[idx]:
                dist[cid, label] += 1
        return dist

    def test_loader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=256,
                          shuffle=False, num_workers=0, pin_memory=False)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FEDERATED CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class FederatedClient:
    """
    Represents one participant in the federation.
    Each client owns a local dataset shard and a model replica.
    """

    def __init__(self, client_id: int, dataset, indices: List[int], cfg: SimConfig):
        self.client_id = client_id
        self.cfg       = cfg
        self.device    = cfg.device

        self.local_dataset  = Subset(dataset, indices)
        self.local_loader   = DataLoader(
            self.local_dataset, batch_size=cfg.batch_size,
            shuffle=True, num_workers=0, pin_memory=False,
            drop_last=True if len(indices) >= cfg.batch_size else False,
        )
        self.n_samples = len(indices)

        # Local model replica (will be overwritten by server weights each round)
        self.model = build_model(cfg).to(self.device)

        # History
        self.train_losses: List[float] = []
        self.train_accs:   List[float] = []

    # ── Receive global model ────────────────────────────────────────────────
    def receive_global_model(self, global_state: dict):
        """Download global weights from server → local model."""
        self.model.load_state_dict(copy.deepcopy(global_state))

    # ── Local training ──────────────────────────────────────────────────────
    def train(self, global_state: Optional[dict] = None) -> dict:
        """
        Train locally for cfg.local_epochs epochs.
        Returns: dict with updated weights and training stats.
        """
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            momentum=0.9,
            weight_decay=self.cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        epoch_losses, epoch_accs = [], []

        for epoch in range(self.cfg.local_epochs):
            batch_losses, correct, total = [], 0, 0

            for X, y in self.local_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X)
                loss   = criterion(logits, y)

                # FedProx: add proximal term relative to global model
                if self.cfg.aggregation == "fedprox" and global_state is not None:
                    prox = 0.0
                    for name, param in self.model.named_parameters():
                        if name in global_state:
                            g = global_state[name].to(self.device)
                            prox += ((param - g) ** 2).sum()
                    loss = loss + (self.cfg.fedprox_mu / 2) * prox

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                batch_losses.append(loss.item())
                preds    = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)

            avg_loss = float(np.mean(batch_losses))
            acc      = correct / max(total, 1)
            epoch_losses.append(avg_loss)
            epoch_accs.append(acc)

            if self.cfg.verbose_clients:
                info(f"    Client {self.client_id:02d} | epoch {epoch+1}/{self.cfg.local_epochs}"
                     f" | loss {avg_loss:.4f} | acc {acc:.3f}")

        self.train_losses.append(float(np.mean(epoch_losses)))
        self.train_accs.append(float(np.mean(epoch_accs)))

        return {
            "state_dict": copy.deepcopy(self.model.state_dict()),
            "n_samples":  self.n_samples,
            "loss":       float(np.mean(epoch_losses)),
            "acc":        float(np.mean(epoch_accs)),
        }

    # ── Local evaluation ────────────────────────────────────────────────────
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate current local model on local data."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in self.local_loader:
                X, y = X.to(self.device), y.to(self.device)
                out  = self.model(X)
                total_loss += criterion(out, y).item() * y.size(0)
                correct    += (out.argmax(1) == y).sum().item()
                total      += y.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)

    def __repr__(self):
        return f"<Client {self.client_id:02d} | {self.n_samples} samples>"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FEDERATED SERVER
# ══════════════════════════════════════════════════════════════════════════════

class FederatedServer:
    """
    Central coordinator: holds global model, selects clients each round,
    broadcasts weights, and aggregates updates.
    """

    def __init__(self, global_model: nn.Module, clients: List[FederatedClient],
                 cfg: SimConfig):
        self.global_model = global_model.to(cfg.device)
        self.clients      = clients
        self.cfg          = cfg
        self.device       = cfg.device
        self.round        = 0

        # FedAdam state
        self._adam_m: Optional[dict] = None
        self._adam_v: Optional[dict] = None

        # History
        self.global_acc:  List[float] = []
        self.global_loss: List[float] = []
        self.round_times: List[float] = []

    @property
    def global_state(self) -> dict:
        return self.global_model.state_dict()

    # ── Client selection ────────────────────────────────────────────────────
    def select_clients(self) -> List[FederatedClient]:
        k = max(1, int(self.cfg.clients_per_round * len(self.clients)))
        return random.sample(self.clients, k)

    # ── Broadcast ───────────────────────────────────────────────────────────
    def broadcast(self, selected: List[FederatedClient]):
        step("SERVER → CLIENTS", f"Broadcasting global model to {len(selected)} clients …")
        for client in selected:
            client.receive_global_model(self.global_state)
        success(f"Global model delivered to clients: {[c.client_id for c in selected]}")

    # ── Collect & aggregate ─────────────────────────────────────────────────
    def aggregate(self, updates: List[dict]):
        method = self.cfg.aggregation
        step("CLIENTS → SERVER", f"Receiving {len(updates)} weight updates | Aggregating via {method.upper()} …")

        if method in ("fedavg", "fedprox"):
            self._fedavg(updates)
        elif method == "fedmedian":
            self._fedmedian(updates)
        elif method == "fedadam":
            self._fedadam(updates)
        else:
            raise ValueError(f"Unknown aggregation: {method}")

        success("Global model updated.")

    def _fedavg(self, updates: List[dict]):
        total = sum(u["n_samples"] for u in updates)
        new_state = copy.deepcopy(updates[0]["state_dict"])
        for key in new_state:
            new_state[key] = torch.zeros_like(new_state[key], dtype=torch.float32)
            for u in updates:
                w = u["n_samples"] / total
                new_state[key] += w * u["state_dict"][key].float().to(self.device)
        self.global_model.load_state_dict(new_state)

    def _fedmedian(self, updates: List[dict]):
        new_state = {}
        for key in updates[0]["state_dict"]:
            stacked = torch.stack([u["state_dict"][key].float().to(self.device)
                                   for u in updates])
            new_state[key] = stacked.median(dim=0).values
        self.global_model.load_state_dict(new_state)

    def _fedadam(self, updates: List[dict]):
        """Server-side Adam on the pseudo-gradient from FedAvg delta."""
        total = sum(u["n_samples"] for u in updates)

        # Compute weighted average of client models (= FedAvg result)
        avg_state = copy.deepcopy(updates[0]["state_dict"])
        for key in avg_state:
            avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)
            for u in updates:
                w = u["n_samples"] / total
                avg_state[key] += w * u["state_dict"][key].float().to(self.device)

        # Pseudo-gradient: delta = avg_client - global
        delta = {k: avg_state[k] - self.global_state[k].float()
                 for k in avg_state}

        # Init Adam moments on first call
        if self._adam_m is None:
            self._adam_m = {k: torch.zeros_like(v) for k, v in delta.items()}
            self._adam_v = {k: torch.zeros_like(v) for k, v in delta.items()}

        b1, b2 = self.cfg.fedadam_beta1, self.cfg.fedadam_beta2
        lr     = self.cfg.fedadam_lr
        eps    = 1e-8
        t      = self.round + 1

        new_state = copy.deepcopy(self.global_state)
        for key in delta:
            self._adam_m[key] = b1 * self._adam_m[key] + (1 - b1) * delta[key]
            self._adam_v[key] = b2 * self._adam_v[key] + (1 - b2) * delta[key] ** 2
            m_hat = self._adam_m[key] / (1 - b1 ** t)
            v_hat = self._adam_v[key] / (1 - b2 ** t)
            new_state[key] = (new_state[key].float()
                              + lr * m_hat / (v_hat.sqrt() + eps)).to(new_state[key].dtype)
        self.global_model.load_state_dict(new_state)

    # ── Global evaluation ───────────────────────────────────────────────────
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                out  = self.global_model(X)
                total_loss += criterion(out, y).item() * y.size(0)
                correct    += (out.argmax(1) == y).sum().item()
                total      += y.size(0)
        loss = total_loss / max(total, 1)
        acc  = correct    / max(total, 1)
        self.global_loss.append(loss)
        self.global_acc.append(acc)
        return loss, acc

    def per_class_accuracy(self, test_loader: DataLoader, num_classes: int) -> np.ndarray:
        """Compute accuracy per class for the global model."""
        self.global_model.eval()
        correct = np.zeros(num_classes)
        total   = np.zeros(num_classes)
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.global_model(X).argmax(1).cpu().numpy()
                y_np  = y.cpu().numpy()
                for pred, true in zip(preds, y_np):
                    total[true]   += 1
                    if pred == true:
                        correct[true] += 1
        return correct / np.maximum(total, 1)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FEDERATED SIMULATION (MAIN ORCHESTRATOR)
# ══════════════════════════════════════════════════════════════════════════════

class FederatedSimulation:
    """
    Main simulation orchestrator.
    Creates server, spawns clients, runs the FL loop, collects metrics, plots.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        _seed(cfg.seed)

        os.makedirs(cfg.output_dir, exist_ok=True)

        # Build all objects
        header("INITIALISING FEDERATION")
        step("DATA", f"Loading {cfg.dataset.upper()} …")
        self.distributor = DataDistributor(cfg)
        self.test_loader = self.distributor.test_loader()
        success(f"Dataset loaded | Train: {len(self.distributor.train_dataset)} "
                f"| Test: {len(self.distributor.test_dataset)}")

        step("DATA", f"Splitting across {cfg.num_clients} clients [{cfg.distribution}] …")
        self.client_indices = self.distributor.split()
        label_dist = self.distributor.label_distribution(self.client_indices)
        info(f"Mean samples/client: {np.mean([len(i) for i in self.client_indices]):.0f} "
             f"| Min: {min(len(i) for i in self.client_indices)} "
             f"| Max: {max(len(i) for i in self.client_indices)}")

        step("CLIENTS", f"Instantiating {cfg.num_clients} FederatedClient objects …")
        self.clients = [
            FederatedClient(cid, self.distributor.train_dataset, idx, cfg)
            for cid, idx in enumerate(self.client_indices)
        ]
        for c in self.clients:
            info(str(c))

        step("SERVER", "Building global model + FederatedServer …")
        global_model = build_model(cfg)
        self.server  = FederatedServer(global_model, self.clients, cfg)
        success(f"Server ready | model: {cfg.model.upper()} "
                f"| aggregation: {cfg.aggregation.upper()}")

        # Metric storage
        self.round_metrics: List[Dict] = []
        self.label_dist     = label_dist

    # ── Main loop ────────────────────────────────────────────────────────────
    def run(self):
        cfg = self.cfg
        cfg.summary()

        header("STARTING FEDERATED TRAINING LOOP")
        print(f"  {BOLD}{cfg.num_rounds} rounds × {cfg.local_epochs} local epochs "
              f"× {int(cfg.num_clients * cfg.clients_per_round)} clients/round{RESET}\n")

        total_start = time.time()

        for rnd in range(1, cfg.num_rounds + 1):
            self.server.round = rnd - 1
            t_start = time.time()

            # ─── Banner ────────────────────────────────────────────────────
            print(f"\n{BOLD}{BLUE}  ┌── ROUND {rnd:02d}/{cfg.num_rounds:02d} "
                  f"{'─'*50}┐{RESET}")

            # 1. Select clients
            selected = self.server.select_clients()
            step("SELECT", f"Selected clients: {[c.client_id for c in selected]}")

            # 2. Broadcast global model
            self.server.broadcast(selected)

            # 3. Local training on each selected client
            step("TRAIN", f"Clients training for {cfg.local_epochs} local epoch(s) …")
            updates = []
            client_losses, client_accs = [], []

            for client in selected:
                upd = client.train(
                    global_state=self.server.global_state
                    if cfg.aggregation == "fedprox" else None
                )
                updates.append(upd)
                client_losses.append(upd["loss"])
                client_accs.append(upd["acc"])
                info(f"    ↳ Client {client.client_id:02d} | "
                     f"loss: {upd['loss']:.4f} | acc: {upd['acc']:.3f} | "
                     f"samples: {client.n_samples}")

            avg_client_loss = float(np.mean(client_losses))
            avg_client_acc  = float(np.mean(client_accs))

            # 4. Clients send weights → Server aggregates
            self.server.aggregate(updates)

            # 5. Global evaluation
            step("EVAL", "Evaluating global model on test set …")
            g_loss, g_acc = self.server.evaluate(self.test_loader)
            t_elapsed = time.time() - t_start
            self.server.round_times.append(t_elapsed)

            # ─── Round summary ─────────────────────────────────────────────
            print(f"  {BOLD}  ┌─ ROUND {rnd:02d} SUMMARY {'─'*46}┐{RESET}")
            print(f"  {BOLD}  │{RESET}  Global  → Loss: {_c(CYAN, f'{g_loss:.4f}')}"
                  f"  |  Acc: {_c(GREEN, f'{g_acc*100:.2f}%')}")
            print(f"  {BOLD}  │{RESET}  Clients → Avg Loss: {_c(YELLOW, f'{avg_client_loss:.4f}')}"
                  f"  |  Avg Acc: {_c(YELLOW, f'{avg_client_acc*100:.2f}%')}")
            print(f"  {BOLD}  │{RESET}  Round time: {_c(DIM, f'{t_elapsed:.1f}s')}")
            print(f"  {BOLD}  └{'─'*62}┘{RESET}")

            self.round_metrics.append({
                "round":            rnd,
                "global_loss":      g_loss,
                "global_acc":       g_acc,
                "avg_client_loss":  avg_client_loss,
                "avg_client_acc":   avg_client_acc,
                "round_time":       t_elapsed,
                "n_selected":       len(selected),
            })

        total_time = time.time() - total_start
        header("TRAINING COMPLETE")
        best_acc = max(m["global_acc"] for m in self.round_metrics)
        best_rnd = next(m["round"] for m in self.round_metrics if m["global_acc"] == best_acc)
        print(f"  Total time     : {total_time:.1f}s")
        print(f"  Best global acc: {_c(GREEN, f'{best_acc*100:.2f}%')} (round {best_rnd})")
        final_acc = self.round_metrics[-1]["global_acc"] * 100
        print(f"  Final global acc: {_c(GREEN, f'{final_acc:.2f}%')}")

        # Per-class accuracy
        self.per_class_acc = self.server.per_class_accuracy(
            self.test_loader, self.distributor.num_classes
        )

        return self.round_metrics

    # ── Plotting ─────────────────────────────────────────────────────────────
    def plot_results(self, save: bool = True, show: bool = False):
        header("GENERATING PLOTS")
        cfg = self.cfg
        metrics = self.round_metrics
        rounds  = [m["round"] for m in metrics]

        # ── Palette ──────────────────────────────────────────────────────────
        DARK_BG   = "#0d1117"
        CARD_BG   = "#161b22"
        GRID_CLR  = "#21262d"
        ACC_CLR   = "#58a6ff"
        LOSS_CLR  = "#ff7b72"
        CL_ACC    = "#3fb950"
        CL_LOSS   = "#d29922"
        TIME_CLR  = "#bc8cff"
        TEXT_CLR  = "#c9d1d9"
        TITLE_CLR = "#f0f6fc"

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(20, 14), facecolor=DARK_BG)
        fig.suptitle(
            f"Federated Learning · {cfg.dataset.upper()} · {cfg.model.upper()} · "
            f"{cfg.aggregation.upper()} · {cfg.distribution}  "
            f"(N={cfg.num_clients} clients, {cfg.num_rounds} rounds)",
            fontsize=15, fontweight="bold", color=TITLE_CLR, y=0.98,
        )
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

        def _ax(pos, title):
            ax = fig.add_subplot(pos)
            ax.set_facecolor(CARD_BG)
            ax.tick_params(colors=TEXT_CLR, labelsize=9)
            ax.spines[:].set_color(GRID_CLR)
            ax.grid(True, color=GRID_CLR, linewidth=0.6, alpha=0.7)
            ax.set_title(title, color=TITLE_CLR, fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel("Round", color=TEXT_CLR, fontsize=9)
            return ax

        # ── 1. Global Accuracy ───────────────────────────────────────────────
        ax1 = _ax(gs[0, 0], "Global Model Accuracy")
        g_accs = [m["global_acc"] * 100 for m in metrics]
        ax1.plot(rounds, g_accs, color=ACC_CLR, lw=2.5, marker="o", ms=5, label="Global Acc")
        ax1.axhline(max(g_accs), color=ACC_CLR, ls="--", lw=1, alpha=0.5,
                    label=f"Peak {max(g_accs):.2f}%")
        ax1.fill_between(rounds, g_accs, alpha=0.12, color=ACC_CLR)
        ax1.set_ylabel("Accuracy (%)", color=TEXT_CLR, fontsize=9)
        ax1.legend(fontsize=8, labelcolor=TEXT_CLR, framealpha=0.2)

        # ── 2. Global Loss ───────────────────────────────────────────────────
        ax2 = _ax(gs[0, 1], "Global Model Loss")
        g_losses = [m["global_loss"] for m in metrics]
        ax2.plot(rounds, g_losses, color=LOSS_CLR, lw=2.5, marker="s", ms=5)
        ax2.fill_between(rounds, g_losses, alpha=0.12, color=LOSS_CLR)
        ax2.set_ylabel("Cross-Entropy Loss", color=TEXT_CLR, fontsize=9)

        # ── 3. Avg Client Train Accuracy ─────────────────────────────────────
        ax3 = _ax(gs[0, 2], "Avg Client Train Accuracy")
        cl_accs = [m["avg_client_acc"] * 100 for m in metrics]
        ax3.plot(rounds, cl_accs, color=CL_ACC, lw=2.5, marker="^", ms=5)
        ax3.fill_between(rounds, cl_accs, alpha=0.12, color=CL_ACC)
        ax3.set_ylabel("Accuracy (%)", color=TEXT_CLR, fontsize=9)

        # ── 4. Client vs Global Accuracy overlay ─────────────────────────────
        ax4 = _ax(gs[1, 0], "Client Train vs Global Test Accuracy")
        ax4.plot(rounds, g_accs,  color=ACC_CLR, lw=2, label="Global (test)")
        ax4.plot(rounds, cl_accs, color=CL_ACC,  lw=2, label="Avg Client (train)", ls="--")
        ax4.set_ylabel("Accuracy (%)", color=TEXT_CLR, fontsize=9)
        ax4.legend(fontsize=8, labelcolor=TEXT_CLR, framealpha=0.2)

        # ── 5. Avg Client Train Loss ──────────────────────────────────────────
        ax5 = _ax(gs[1, 1], "Avg Client Train Loss")
        cl_losses = [m["avg_client_loss"] for m in metrics]
        ax5.plot(rounds, cl_losses, color=CL_LOSS, lw=2.5, marker="D", ms=4)
        ax5.fill_between(rounds, cl_losses, alpha=0.12, color=CL_LOSS)
        ax5.set_ylabel("Loss", color=TEXT_CLR, fontsize=9)

        # ── 6. Round Time ─────────────────────────────────────────────────────
        ax6 = _ax(gs[1, 2], "Round Wall-Clock Time")
        rtimes = [m["round_time"] for m in metrics]
        bars   = ax6.bar(rounds, rtimes, color=TIME_CLR, alpha=0.7, width=0.6)
        ax6.plot(rounds, rtimes, color=TIME_CLR, lw=1.5, ls="--", alpha=0.8)
        ax6.set_ylabel("Seconds", color=TEXT_CLR, fontsize=9)

        # ── 7. Per-Class Accuracy ─────────────────────────────────────────────
        ax7 = _ax(gs[2, 0], "Final Per-Class Accuracy")
        n_cls   = self.distributor.num_classes
        colors7 = plt.cm.plasma(np.linspace(0.2, 0.9, n_cls))
        bars7   = ax7.bar(range(n_cls), self.per_class_acc * 100,
                          color=colors7, alpha=0.85, width=0.7)
        ax7.set_xticks(range(n_cls))
        ax7.set_xticklabels([str(i) for i in range(n_cls)],
                             color=TEXT_CLR, fontsize=8)
        ax7.set_xlabel("Class", color=TEXT_CLR, fontsize=9)
        ax7.set_ylabel("Accuracy (%)", color=TEXT_CLR, fontsize=9)
        ax7.axhline(np.mean(self.per_class_acc) * 100, color="#fff", ls="--",
                    lw=1, alpha=0.5, label=f"Mean {np.mean(self.per_class_acc)*100:.1f}%")
        ax7.legend(fontsize=8, labelcolor=TEXT_CLR, framealpha=0.2)

        # ── 8. Data Distribution Heatmap ──────────────────────────────────────
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.set_facecolor(CARD_BG)
        ax8.set_title("Data Distribution (Clients × Classes)",
                       color=TITLE_CLR, fontsize=11, fontweight="bold", pad=8)
        im = ax8.imshow(
            self.label_dist, aspect="auto",
            cmap="Blues" if cfg.distribution == "iid" else "YlOrRd",
        )
        ax8.set_xlabel("Class", color=TEXT_CLR, fontsize=9)
        ax8.set_ylabel("Client", color=TEXT_CLR, fontsize=9)
        ax8.tick_params(colors=TEXT_CLR, labelsize=8)
        ax8.spines[:].set_color(GRID_CLR)
        plt.colorbar(im, ax=ax8, label="# samples").ax.yaxis.label.set_color(TEXT_CLR)

        # ── 9. Summary Stats Card ─────────────────────────────────────────────
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.set_facecolor(CARD_BG)
        ax9.axis("off")
        ax9.set_title("Summary Statistics", color=TITLE_CLR,
                       fontsize=11, fontweight="bold", pad=8)

        conv_round = next((m["round"] for m in metrics
                           if m["global_acc"] >= 0.8), "N/A")
        stats = [
            ("Best Global Acc",   f"{max(g_accs):.2f}%"),
            ("Final Global Acc",  f"{g_accs[-1]:.2f}%"),
            ("Best Client Acc",   f"{max(cl_accs):.2f}%"),
            ("Min Global Loss",   f"{min(g_losses):.4f}"),
            ("Avg Round Time",    f"{np.mean(rtimes):.1f}s"),
            ("Total Wall Time",   f"{sum(rtimes):.1f}s"),
            ("Conv. Round (≥80%)", str(conv_round)),
            ("Aggregation",       cfg.aggregation.upper()),
            ("Distribution",      cfg.distribution),
            ("Device",            str(cfg.device).upper()),
        ]
        for i, (k, v) in enumerate(stats):
            y_pos = 0.95 - i * 0.092
            ax9.text(0.03, y_pos, k + ":", transform=ax9.transAxes,
                     color=TEXT_CLR, fontsize=9, va="top")
            ax9.text(0.62, y_pos, v, transform=ax9.transAxes,
                     color=ACC_CLR, fontsize=9, va="top", fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save:
            out = os.path.join(cfg.output_dir,
                               f"fl_{cfg.dataset}_{cfg.model}_{cfg.aggregation}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            success(f"Plot saved → {out}")

        if show:
            plt.show()

        plt.close(fig)
        return out if save else None

    # ── Print full metrics table ──────────────────────────────────────────────
    def print_metrics_table(self):
        header("METRICS TABLE")
        cols = f"{'Rnd':>4}  {'G-Acc%':>7}  {'G-Loss':>7}  {'CL-Acc%':>8}  {'CL-Loss':>8}  {'Time(s)':>7}"
        print(f"  {BOLD}{cols}{RESET}")
        divider()
        for m in self.round_metrics:
            print(f"  {m['round']:>4}  "
                  f"{m['global_acc']*100:>7.2f}  "
                  f"{m['global_loss']:>7.4f}  "
                  f"{m['avg_client_acc']*100:>8.2f}  "
                  f"{m['avg_client_loss']:>8.4f}  "
                  f"{m['round_time']:>7.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Federated Learning Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",        default="mnist",
                   choices=["mnist", "fashionmnist", "cifar10"])
    p.add_argument("--model",          default="cnn",
                   choices=["mlp", "cnn", "resnet_lite"])
    p.add_argument("--num_clients",    type=int,   default=10)
    p.add_argument("--clients_per_round", type=float, default=0.5)
    p.add_argument("--num_rounds",     type=int,   default=15)
    p.add_argument("--local_epochs",   type=int,   default=3)
    p.add_argument("--lr",             type=float, default=0.01)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--aggregation",    default="fedavg",
                   choices=["fedavg", "fedprox", "fedmedian", "fedadam"])
    p.add_argument("--distribution",   default="iid",
                   choices=["iid", "non_iid_dirichlet", "pathological"])
    p.add_argument("--alpha",          type=float, default=0.5,
                   help="Dirichlet alpha (smaller → more heterogeneous)")
    p.add_argument("--shards",         type=int,   default=2,
                   help="Classes per client for pathological split")
    p.add_argument("--fedprox_mu",     type=float, default=0.01)
    p.add_argument("--no_gpu",         action="store_true")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--output_dir",     default="./fl_results")
    p.add_argument("--verbose_clients",action="store_true")
    p.add_argument("--show_plot",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = SimConfig(
        dataset            = args.dataset,
        model              = args.model,
        num_clients        = args.num_clients,
        clients_per_round  = args.clients_per_round,
        num_rounds         = args.num_rounds,
        local_epochs       = args.local_epochs,
        learning_rate      = args.lr,
        batch_size         = args.batch_size,
        aggregation        = args.aggregation,
        distribution       = args.distribution,
        dirichlet_alpha    = args.alpha,
        shards_per_client  = args.shards,
        fedprox_mu         = args.fedprox_mu,
        use_gpu            = not args.no_gpu,
        seed               = args.seed,
        output_dir         = args.output_dir,
        verbose_clients    = args.verbose_clients,
    )

    sim = FederatedSimulation(cfg)

    # Show model size
    model_tmp = build_model(cfg)
    step("MODEL", f"Parameters: {count_parameters(model_tmp):,}")

    sim.run()
    sim.print_metrics_table()
    sim.plot_results(save=True, show=args.show_plot)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PROGRAMMATIC API (for Jupyter / scripting)
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    dataset="mnist", model="cnn",
    num_clients=10, clients_per_round=0.5,
    num_rounds=15, local_epochs=3,
    learning_rate=0.01, batch_size=32,
    aggregation="fedavg",
    distribution="iid",
    dirichlet_alpha=0.5,
    use_gpu=True, seed=42,
    output_dir="./fl_results",
    **kwargs,
):
    """
    Convenience wrapper for programmatic use.

    Example
    -------
    >>> from federated_learning import run_experiment
    >>> metrics = run_experiment(
    ...     dataset="cifar10", model="resnet_lite",
    ...     num_clients=20, num_rounds=30,
    ...     aggregation="fedadam",
    ...     distribution="non_iid_dirichlet", dirichlet_alpha=0.1,
    ... )
    """
    cfg = SimConfig(
        dataset=dataset, model=model,
        num_clients=num_clients, clients_per_round=clients_per_round,
        num_rounds=num_rounds, local_epochs=local_epochs,
        learning_rate=learning_rate, batch_size=batch_size,
        aggregation=aggregation,
        distribution=distribution, dirichlet_alpha=dirichlet_alpha,
        use_gpu=use_gpu, seed=seed,
        output_dir=output_dir,
        **{k: v for k, v in kwargs.items() if hasattr(SimConfig, k)},
    )
    sim = FederatedSimulation(cfg)
    sim.run()
    sim.print_metrics_table()
    plot_path = sim.plot_results(save=True, show=False)
    return sim.round_metrics, plot_path


if __name__ == "__main__":
    main()
