# PyTorch Non-Linear Regression Model (\(x^2\) Approximator)

This repository contains a deep learning model built to approximate the quadratic function \(y = x^2\). It demonstrates how a multi-layer neural network can learn non-linear functions from synthetic, noisy datasets.

## Model Hyperparameters
* **Hidden Layers:** 4 fully connected layers.
* **Layer Width:** 128 nodes per hidden layer.
* **Activation:** LeakyReLU (\(alpha = 0.25\)).
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
   git clone https://github.com/maverickmhooper-arch/AI-Polynomial-Regression.git
   cd AI-Polynomial-Regression
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
python Polynomial_Regression.py
```

### CLI Options Available
* `[T]rain`: Trains the model for up to 20,000 epochs. Includes automated early stopping based on Exponential Moving Average (EMA) loss thresholds.
* `[I]nference`: Test the trained model against true mathematical outputs by entering any number between \(-50\) and \(50\).
* `[Q]uit`: Exits the loop and terminates the program.

---

## Code Highlight: Data Generation
The code allows for comparisons of the model's progress across different time frames. It takes snippets from different points in the training, and compiles them into 4 different graphs that can be easily compared. 

### What I Learned

In working on this project, I learned that learning models like this are incredible good at finding patterns in exponential functions. It was difficult to find where to start on this, but after I got a foothold in the basics, I just grew from there.
# I learned the following
* 1. How to use the Adam optimizer and refine the learning rate.
* 2. How to create an inference interface so the user can ask for an input-output.
* 3. Why more depth and layers of a model helps the model reach the correct answer faster.
* 4. I learned how to use matplotlib to model progress from the network.

### Hardest Part
The hardest part was figuring out how to get the returned model from the training_model() function and bringing it into the inference part. Also, it was very tedious doing the floats and input_vals, but that's just me.

## The Future
In the future, I'm probably going to add some normalization and a scikit-learn model for a comparison against another model. I'm also going to move toward multi-variable in the future. 

If you've read to the end, you get a cookie! 🍪
