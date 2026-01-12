import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from model import MLP
from colored_mnist import get_colored_mnist
from brightness_mnist import get_brightness_mnist as get_colored_mnist


# IRM penalty: gradient of the loss w.r.t. a scalar
def compute_irm_penalty(logits, labels):
    scale = torch.tensor(1.).requires_grad_()
    loss = nn.BCEWithLogitsLoss()(logits * scale, labels)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)

def train_irm(epochs=501, lr=0.001, lambda_penalty=1000, penalty_anneal_iters=100, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training and test environments
    environments, test_data = get_colored_mnist()
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Accuracy tracking lists
    env1_acc_list = []
    env2_acc_list = []
    env3_acc_list = []
    test_acc_list = []
    steps = []

    for step in range(epochs):
        env_losses = []
        env_penalties = []

        # Compute loss and penalty per environment
        for env_images, env_labels in environments:
            env_images, env_labels = env_images.to(device), env_labels.to(device)
            logits = model(env_images).squeeze()

            loss = nn.BCEWithLogitsLoss()(logits, env_labels)
            penalty = compute_irm_penalty(logits, env_labels)

            env_losses.append(loss)
            env_penalties.append(penalty)

        total_loss = torch.stack(env_losses).mean()
        total_penalty = torch.stack(env_penalties).mean()

        # Apply stronger penalty after a warm-up period
        penalty_weight = lambda_penalty if step >= penalty_anneal_iters else 1.0
        final_loss = total_loss + penalty_weight * total_penalty

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # Evaluation every 50 steps
        if step % 50 == 0:
            with torch.no_grad():
                accs = []
                #for name, (x_env, y_env) in zip(["env1", "env2", "test"], environments + [test_data]):
                for name, (x_env, y_env) in zip(["env1", "env2", "env3", "test"], environments + [test_data]):
                    x_env, y_env = x_env.to(device), y_env.to(device)
                    preds = (torch.sigmoid(model(x_env).squeeze()) > 0.5).float()
                    acc = (preds == y_env).float().mean().item()
                    accs.append(acc)

                env1_acc_list.append(accs[0])
                env2_acc_list.append(accs[1])
                env3_acc_list.append(accs[2])
                test_acc_list.append(accs[3])
                steps.append(step)

                print(f"Step {step}: Env1={accs[0]:.3f}, Env2={accs[1]:.3f}, Env3={accs[2]:.3f}, Test={accs[3]:.3f}")
    # Plot after training
    plt.figure(figsize=(8, 5))
    plt.plot(steps, env1_acc_list, label="Env1 Accuracy")
    plt.plot(steps, env2_acc_list, label="Env2 Accuracy")
    plt.plot(steps, env3_acc_list, label="Env3 Accuracy")
    plt.plot(steps, test_acc_list, label="Test Accuracy")
    plt.xlabel("Training Step")
    plt.ylabel("Accuracy")
    plt.title("IRM Performance Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("irm_accuracy_plot.png")
    plt.show()

# Run if script is executed
if __name__ == "__main__":
    train_irm()
