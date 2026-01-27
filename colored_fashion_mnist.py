import torch
from torchvision import datasets
import numpy as np

def make_environment(images, labels, color_prob):
    # Flatten and normalize images
    images = images.reshape((-1, 28 * 28)) / 255.
    
    # Binarize Labels
    # FashionMNIST has 10 classes. We group them into 2 super-classes:
    # Class 0: Upper-body clothes (T-shirt, Pullover, Dress, Coat, Shirt) -> Indices [0, 2, 3, 4, 6]
    # Class 1: Footwear & Accessories (Trouser, Sandal, Sneaker, Bag, Ankle boot) -> Indices [1, 5, 7, 8, 9]
    labels = labels.clone() # create a copy to avoid modifying original data
    labels_bin = torch.zeros_like(labels)
    
    # Define mask for Class 1 (Footwear/Accessories)
    mask_class1 = (labels == 1) | (labels == 5) | (labels == 7) | (labels == 8) | (labels == 9)
    labels_bin[mask_class1] = 1
    # Everything else remains 0 (Upper-body clothes)
    labels = labels_bin.float()

    # Flip labels randomly with 25% probability (adds noise)
    labels = torch.where(torch.rand(len(labels)) < 0.25, 1 - labels, labels)

    # Apply color correlation: with probability, color matches label
    color = torch.where(torch.rand(len(labels)) < color_prob, labels, 1 - labels)

    # Expand image to two channels: [R, G]
    images = images.unsqueeze(1).repeat(1, 2, 1).view(-1, 2, 28, 28)

    # Adjust shape of color tensor for broadcasting
    color = color.view(-1, 1, 1)

    # Apply color to the two channels
    images[:, 0] *= color        # Red channel if color = 1
    images[:, 1] *= 1 - color    # Green channel if color = 0

    return images, labels

# Load Colored Fashion MNIST environments
def get_colored_fashion_mnist():
    # Download FashionMNIST (train/test))
    mnist_train = datasets.FashionMNIST(root='data', train=True, download=True)
    mnist_test = datasets.FashionMNIST(root='data', train=False, download=True)

    # Split training data into three parts
    # Env1 -> 90% correlation
    env1_imgs, env1_labels = make_environment(
        mnist_train.data[:20000], mnist_train.targets[:20000], color_prob=0.9
    )
    # Env2 -> 80% correlation
    env2_imgs, env2_labels = make_environment(
        mnist_train.data[20000:40000], mnist_train.targets[20000:40000], color_prob=0.8
    )
    # Env3 -> 70% correlation
    env3_imgs, env3_labels = make_environment(
        mnist_train.data[40000:], mnist_train.targets[40000:], color_prob=0.7
    )

    # Test environment: no correlation (10%)
    test_imgs, test_labels = make_environment(
        mnist_test.data, mnist_test.targets, color_prob=0.1
    )

    return [(env1_imgs, env1_labels), (env2_imgs, env2_labels), (env3_imgs, env3_labels)], (test_imgs, test_labels)