# PyTorch Non-Linear Regression Model (\(x^2\) Approximator)

This repo contains a program which runs a Neural Network and a scikit-learn model which are both trying to converge on a specific polynomial, x^2 in this case. It shows how Neural Networks compare with scikit-learn, as well as how learning rates and error threshols affect the results.

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
   Ensure you have `torch, numpy, time, and sklearn` installed:
   ```bash
   pip install torch
   pip install numpy
   pip install time
   pip install sklearn
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

### Results 
<img width="600" height="500" alt="model (2)" src="https://github.com/user-attachments/assets/bee61b40-f66e-4fcd-8ea1-21c2d7372de4" />

These are results of one run that I ran, showed on matplotlib.

# What I Learned

In working on this project, I learned that learning models like this are incredible good at finding patterns in exponential functions. It was difficult to find where to start on this, but after I got a foothold in the basics, I just grew from there.
### I learned the following
1. How to use the Adam optimizer and refine the learning rate.
2. How to create an inference interface so the user can ask for an input-output.
3. Why more depth and layers of a model helps the model reach the correct answer faster.
4. I learned how to use matplotlib to model progress from the network.
5. I learned how matplotlib can be used to compare different architectures and results.
6. Scikit-learn is much faster and more accurate for simply problems like this. 

### Hardest Part
The hardest part was creating the scikit-learn model and the graph comparisons. This is simply because I don't use scikit-learn often.

## The Future
In the future, I'm probably going to add some normalization and a graph showing the model converging toward the error threshold. I'm also going to move toward multi-variable in the future. 

If you've read to the end, you get a cookie! 🍪
