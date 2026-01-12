import torch
from torchvision import datasets
import numpy as np

# Function to generate one environment with a specific color correlation
def make_environment(images, labels, color_prob):
    # Flatten and normalize images
    images = images.view(-1, 28 * 28) / 255.

    # Convert digits to binary labels: 1 if >=5, else 0
    labels = (labels >= 5).float()

    # Flip labels randomly with 25% probability (adds noise)
    labels = torch.where(torch.rand(len(labels)) < 0.25, 1 - labels, labels)

    # Apply color correlation: with probability, color matches label
    color = torch.where(torch.rand(len(labels)) < color_prob, labels, 1 - labels)

    # Expand image to two channels: [R, G]
    images = images.repeat(1, 2).view(-1, 2, 28, 28)

    # Adjust shape of color tensor for broadcasting
    color = color.view(-1, 1, 1)

    # Apply color to the two channels
    images[:, 0] *= color        # Red channel if color = 1
    images[:, 1] *= 1 - color    # Green channel if color = 0

    return images, labels

# Load Colored MNIST environments
def get_colored_mnist():
    # Download MNIST dataset (train/test)
    mnist_train = datasets.MNIST(root='data', train=True, download=True)
    mnist_test = datasets.MNIST(root='data', train=False, download=True)

    """"
    ## In this part, we reproduce the main results from the paper.

    # First environment: strong correlation (90%)
    env1_imgs, env1_labels = make_environment(
        mnist_train.data[:25000], mnist_train.targets[:25000], color_prob=0.9
    )

    # Second environment: weaker correlation (80%)
    env2_imgs, env2_labels = make_environment(
        mnist_train.data[25000:], mnist_train.targets[25000:], color_prob=0.8
    )
    """
    
    # Here, we'll add a third environment with a different color-label correlation
    
    # Split training data into three parts
    env1_imgs, env1_labels = make_environment(
        mnist_train.data[:16667], mnist_train.targets[:16667], color_prob=0.9
    )
    env2_imgs, env2_labels = make_environment(
        mnist_train.data[16667:33334], mnist_train.targets[16667:33334], color_prob=0.8
    )
    env3_imgs, env3_labels = make_environment(
        mnist_train.data[33334:], mnist_train.targets[33334:], color_prob=0.7
    )


    # Test environment: no correlation (50%)
    test_imgs, test_labels = make_environment(
        mnist_test.data, mnist_test.targets, color_prob=0.5
    )

    return [(env1_imgs, env1_labels), (env2_imgs, env2_labels), (env3_imgs, env3_labels)], (test_imgs, test_labels)
