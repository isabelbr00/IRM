# Invariant Risk Minimization (IRM) vs Empirical Risk Minimization (ERM)

This project demonstrates the difference between **ERM** (Empirical Risk Minimization) and **IRM** (Invariant Risk Minimization) using a synthetic version of the **Colored MNIST** dataset. It follows the approach proposed in the paper [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893).

## Objective

To show how IRM can improve **out-of-distribution (OOD) generalization** compared to ERM, especially when training data contains **spurious correlations** that do not hold in the test environment.

---

## Project Structure

irm-demo/

├── colored_mnist.py # Generates Colored MNIST environments

├── model.py # Simple MLP classifier

├── train_erm.py # Standard ERM training

├── train_irm.py # IRM training with penalty

├── requirements.txt # Required Python packages

└── README.md # Project instructions and description


---

## Installation

### 1. Clone the repository

bash
git clone [https://github.com/yourusername/irm-demo.git](https://github.com/isabelbr00/IRM.git)
cd irm-demo

### 2. Install dependencies

pip install -r requirements.txt
 
---

## Experimental Setup

We create three training environments with different spurious correlations between color and label:

- **Environment 1**: 90% correlation
- **Environment 2**: 80% correlation
- **Environment 3**: 70% correlation

The **test environment** is unbiased (50% correlation). This setup makes IRM’s goal of learning invariant features more evident.

---

## Running the Experiments
Train with ERM:
python train_erm.py

Train with IRM:
python train_irm.py

---

## Experimental Results

### Empirical Risk Minimization (ERM)
The ERM model learns the spurious correlation (color) and fails on the test set.
![ERM Accuracy Plot](erm_accuracy_plot.png)

### Invariant Risk Minimization (IRM)
The IRM model focuses on invariance and maintains stable performance on the test set.
![IRM Accuracy Plot](irm_accuracy_plot.png)

---

## Interpretation

The ERM model quickly learns the spurious correlation and obtains high accuracy in training environments but fails to generalize to the unbiased test environment. 

In contrast, IRM sacrifices some training accuracy in order to learn features that generalize better, resulting in improved accuracy on the unbiased test environment.

---

## References

Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019).
Invariant Risk Minimization
