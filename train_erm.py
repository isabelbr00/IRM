import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from model import MLP
from colored_mnist import get_colored_mnist
from brightness_mnist import get_brightness_mnist as get_colored_mnist


def train_erm(epochs=501, lr=0.001, batch_size=256):
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load two training environments and one test environment
    environments, test_data = get_colored_mnist()

    # Create the model and move it to the device
    model = MLP().to(device)

    # Define optimizer and binary classification loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Merge data from both training environments
    train_imgs = torch.cat([e[0] for e in environments])
    train_labels = torch.cat([e[1] for e in environments])

    # Create a DataLoader for ERM (ignores environments)
    loader = DataLoader(TensorDataset(train_imgs, train_labels), batch_size=batch_size, shuffle=True)

    # Accuracy tracking lists
    env1_acc_list = []
    env2_acc_list = []
    env3_acc_list = []
    test_acc_list = []
    steps = []

    for step in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze()
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
    plt.title("ERM Performance Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("erm_accuracy_plot.png")
    plt.show()

# Run training if this file is executed directly
if __name__ == "__main__":
    train_erm()
