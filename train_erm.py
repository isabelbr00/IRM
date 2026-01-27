import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import TensorDataset, DataLoader
from model import MLP
# Import data generators
from colored_mnist import get_colored_mnist
from colored_fashion_mnist import get_colored_fashion_mnist

def train_erm(args):
    # Reproducibility (Critical for fair comparison with IRM)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--> Training ERM (Standard) on dataset: {args.dataset}")

    # Load the chosen Dataset
    if args.dataset == 'mnist':
        environments, test_data = get_colored_mnist()
    else:
        environments, test_data = get_colored_fashion_mnist()

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Prepare data for ERM (Merge everything!)
    # ERM doesn't care about environments; it just wants to minimize the average error.
    # Therefore, we concatenate data from all training environments into one big pile.
    train_imgs = torch.cat([e[0] for e in environments])
    train_labels = torch.cat([e[1] for e in environments])
    
    # DataLoader to shuffle everything together
    train_loader = DataLoader(
        TensorDataset(train_imgs, train_labels), 
        batch_size=args.batch_size, 
        shuffle=True
    )

    # Lists to track history/metrics
    train_acc_lists = [[] for _ in environments]
    test_acc_list = []
    steps_log = []

    # Training Loop
    step = 0
    while step < args.steps:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x).squeeze()
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluation every 100 steps
            if step % 100 == 0:
                with torch.no_grad():
                    # Test
                    tx, ty = test_data
                    tx, ty = tx.to(device), ty.to(device)
                    t_preds = (torch.sigmoid(model(tx).squeeze()) > 0.5).float()
                    t_acc = (t_preds == ty).float().mean().item()
                    test_acc_list.append(t_acc)
                    
                    # Train
                    log_str = f"Step {step}: "
                    for i, (ex, ey) in enumerate(environments):
                        ex, ey = ex.to(device), ey.to(device)
                        epreds = (torch.sigmoid(model(ex).squeeze()) > 0.5).float()
                        eacc = (epreds == ey).float().mean().item()
                        train_acc_lists[i].append(eacc)
                        log_str += f"Env{i}={eacc:.3f} "
                    
                    log_str += f"| Test={t_acc:.3f}"
                    print(log_str)
                    
                    steps_log.append(step)

            step += 1
            if step >= args.steps:
                break

    # Plot Results
    plt.figure(figsize=(10, 6))
    for i, acc_list in enumerate(train_acc_lists):
        plt.plot(steps_log, acc_list, label=f"Train Env {i} (bias)")
    
    plt.plot(steps_log, test_acc_list, label="Test (OOD)", linewidth=3, color='red', linestyle='--')
    
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title(f"ERM (Standard) on {args.dataset.upper()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filename = f"erm_results_{args.dataset}.png"
    plt.savefig(filename)
    print(f"Graph saved as {filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ERM Demo')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion'], help='Choose: mnist or fashion')
    parser.add_argument('--steps', type=int, default=2001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    train_erm(args)