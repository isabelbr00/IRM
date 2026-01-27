import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import argparse
from model import MLP
# Import data generators
from colored_mnist import get_colored_mnist
from colored_fashion_mnist import get_colored_fashion_mnist


# IRM penalty: gradient of the loss w.r.t. a scalar
def compute_irm_penalty(logits, labels):
    scale = torch.tensor(1.).requires_grad_()
    loss = nn.BCEWithLogitsLoss()(logits * scale, labels)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)

def train_irm(args):
    # Reproducibility
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the chosen Dataset
    print(f"--> Training IRM on dataset: {args.dataset}")
    if args.dataset == 'mnist':
        environments, test_data = get_colored_mnist()
    else:
        environments, test_data = get_colored_fashion_mnist()

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Lists to track history/metrics
    train_acc_lists = [[] for _ in environments]
    test_acc_list = []
    steps = []

    for step in range(args.steps):
        env_losses = []
        env_penalties = []

        for env_imgs, env_labels in environments:
            env_imgs, env_labels = env_imgs.to(device), env_labels.to(device)
            logits = model(env_imgs).squeeze()

            loss = nn.BCEWithLogitsLoss()(logits, env_labels)
            penalty = compute_irm_penalty(logits, env_labels)

            env_losses.append(loss)
            env_penalties.append(penalty)

        total_loss = torch.stack(env_losses).mean()
        total_penalty = torch.stack(env_penalties).mean()

        # Penalty Annealing (Warm-up)
        penalty_weight = args.l2_penalty if step >= args.penalty_anneal_iters else 0.0
        
        final_loss = total_loss + penalty_weight * total_penalty

        optimizer.zero_grad()
        final_loss.backward()
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
                steps.append(step)
                
                # Train
                train_accs_str = ""
                for i, (ex, ey) in enumerate(environments):
                    ex, ey = ex.to(device), ey.to(device)
                    epreds = (torch.sigmoid(model(ex).squeeze()) > 0.5).float()
                    eacc = (epreds == ey).float().mean().item()
                    train_acc_lists[i].append(eacc)
                    train_accs_str += f"Env{i}={eacc:.3f} "

                print(f"Step {step}: {train_accs_str}| Test={t_acc:.3f} | PenaltyWeight={penalty_weight}")

    # Plot Results
    plt.figure(figsize=(10, 6))
    for i, acc_list in enumerate(train_acc_lists):
        plt.plot(steps, acc_list, label=f"Train Env {i} (bias)")
    plt.plot(steps, test_acc_list, label="Test (OOD)", linewidth=3, color='black')
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title(f"IRM on {args.dataset.upper()} (Penalty={args.l2_penalty})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"irm_results_{args.dataset}.png")
    print(f"Graph saved as irm_results_{args.dataset}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IRM Demo')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion'], help='Choose: mnist or fashion')
    parser.add_argument('--steps', type=int, default=2001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_penalty', type=float, default=10000.0, help='Penalty severity IRM')
    parser.add_argument('--penalty_anneal_iters', type=int, default=500, help='Steps before activating the penalty')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    train_irm(args)
