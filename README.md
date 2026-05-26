# PyTorch Non-Linear Regression Model (\(x^2\) Approximator)

This repository contains a deep learning model built to approximate the quadratic function \(y = x^2\). It demonstrates how a multi-layer neural network can learn non-linear functions from synthetic, noisy datasets.

## Model Hyperparameters
* **Hidden Layers:** 4 fully connected layers.
* **Layer Width:** 128 nodes per hidden layer.
* **Activation:** LeakyReLU ((\alpha = 0.25\)).
* **Optimizer:** Adam.
* **Learning Rate:** \(0.0005\) for stable convergence.
* **Data Domain (Span):** \([-50, 50]\).

---

## Architecture Diagram

The model uses a deep dense architecture to map a single input feature to a single continuous output:

```text
Input (1) ──> Linear(128) ──> LeakyReLU ──> Linear(128) ──> LeakyReLU 
          ──> Linear(128) ──> LeakyReLU ──> Linear(128) ──> LeakyReLU ──> Output (1)
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd quadratic-regression
   ```

2. **Install dependencies:**
   Ensure you have `torch` installed:
   ```bash
   pip install torch
   ```

---

## Usage

Run the script from your terminal to launch the interactive command-line interface:

```bash
python main.py
```

### CLI Options Available
* `[T]rain`: Trains the model for up to 20,000 epochs. Includes automated early stopping based on Exponential Moving Average (EMA) loss thresholds.
* `[I]nference`: Test the trained model against true mathematical outputs by entering any number between \(-50\) and \(50\).
* `[Q]uit`: Exits the loop and terminates the program.

---

## Code Highlight: Data Generation

The target values include small Gaussian noise ((\sigma = 0.1\)) to simulate real-world data collection:

```python
# From vector() function
x = torch.linspace(-50, 50, num_generate).view(-1, 1)
y = x**2 + torch.randn(x.size()) * 0.1
```
