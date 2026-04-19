"""
federated_examples.py — Example experiments using the FL framework.

Run any of the examples below individually.
Each experiment prints live step-by-step updates and saves a plot to ./fl_results/.
"""

from federated_learning import run_experiment, SimConfig, FederatedSimulation


# ─────────────────────────────────────────────────────────────────────────────
# Example 1 — Quick sanity check (MNIST, IID, FedAvg, 5 rounds)
# ─────────────────────────────────────────────────────────────────────────────
def example_quick():
    metrics, plot = run_experiment(
        dataset="mnist",
        model="cnn",
        num_clients=5,
        clients_per_round=1.0,
        num_rounds=5,
        local_epochs=2,
        learning_rate=0.01,
        aggregation="fedavg",
        distribution="iid",
    )
    print(f"\nFinal accuracy: {metrics[-1]['global_acc']*100:.2f}%")
    print(f"Plot saved at: {plot}")


# ─────────────────────────────────────────────────────────────────────────────
# Example 2 — Non-IID Dirichlet split (heterogeneous clients)
# ─────────────────────────────────────────────────────────────────────────────
def example_non_iid():
    metrics, plot = run_experiment(
        dataset="mnist",
        model="cnn",
        num_clients=10,
        num_rounds=20,
        local_epochs=5,
        learning_rate=0.01,
        aggregation="fedavg",
        distribution="non_iid_dirichlet",
        dirichlet_alpha=0.1,   # low alpha = very heterogeneous
    )


# ─────────────────────────────────────────────────────────────────────────────
# Example 3 — FedProx on pathological split
# ─────────────────────────────────────────────────────────────────────────────
def example_fedprox():
    metrics, plot = run_experiment(
        dataset="fashionmnist",
        model="cnn",
        num_clients=10,
        num_rounds=15,
        local_epochs=5,
        learning_rate=0.01,
        aggregation="fedprox",
        fedprox_mu=0.05,
        distribution="pathological",
        shards_per_client=2,   # each client sees only 2 classes
    )


# ─────────────────────────────────────────────────────────────────────────────
# Example 4 — CIFAR-10 with ResNet-lite and FedAdam (server-side Adam)
# ─────────────────────────────────────────────────────────────────────────────
def example_cifar_fedadam():
    metrics, plot = run_experiment(
        dataset="cifar10",
        model="resnet_lite",
        num_clients=20,
        clients_per_round=0.4,
        num_rounds=30,
        local_epochs=3,
        learning_rate=0.05,
        batch_size=64,
        aggregation="fedadam",
        fedadam_lr=1e-3,
        distribution="non_iid_dirichlet",
        dirichlet_alpha=0.5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Example 5 — FedMedian (robust to poisoning)
# ─────────────────────────────────────────────────────────────────────────────
def example_fedmedian():
    metrics, plot = run_experiment(
        dataset="mnist",
        model="mlp",
        num_clients=10,
        num_rounds=15,
        local_epochs=4,
        learning_rate=0.02,
        aggregation="fedmedian",
        distribution="iid",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Example 6 — Advanced: use SimConfig + FederatedSimulation directly
# ─────────────────────────────────────────────────────────────────────────────
def example_advanced():
    cfg = SimConfig(
        dataset            = "cifar10",
        model              = "cnn",
        num_clients        = 15,
        clients_per_round  = 0.6,
        num_rounds         = 25,
        local_epochs       = 5,
        learning_rate      = 0.01,
        batch_size         = 64,
        aggregation        = "fedavg",
        distribution       = "non_iid_dirichlet",
        dirichlet_alpha    = 0.3,
        use_gpu            = True,
        seed               = 123,
        verbose_clients    = True,   # print per-epoch client stats
        output_dir         = "./fl_results",
    )

    sim = FederatedSimulation(cfg)
    sim.run()
    sim.print_metrics_table()
    sim.plot_results(save=True, show=False)

    # Access raw per-round data
    for m in sim.round_metrics:
        print(f"Round {m['round']:2d}: "
              f"global_acc={m['global_acc']:.4f}  "
              f"global_loss={m['global_loss']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    examples = {
        "1": ("Quick MNIST sanity check",             example_quick),
        "2": ("Non-IID Dirichlet (MNIST)",             example_non_iid),
        "3": ("FedProx + Pathological (FashionMNIST)", example_fedprox),
        "4": ("CIFAR-10 + ResNet-lite + FedAdam",      example_cifar_fedadam),
        "5": ("FedMedian robust aggregation",           example_fedmedian),
        "6": ("Advanced: direct SimConfig API",         example_advanced),
    }

    print("\n  Choose an example to run:")
    for k, (desc, _) in examples.items():
        print(f"    [{k}] {desc}")

    choice = input("\n  Enter number (default 1): ").strip() or "1"
    _, fn = examples.get(choice, ("", example_quick))
    fn()
