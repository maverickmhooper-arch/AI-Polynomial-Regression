# PyTorch Polynomial Regression Model (\(x^2\) Approximator)

This repo contains a program which runs a Neural Network and a scikit-learn model which are both trying to converge on a specific polynomial, x^2 in this case. It shows how Neural Networks compare with scikit-learn, as well as how learning rates and error thresholds affect the results. It also contains graphs of performance, with one showing the loss rate improving as a function of time, and the other showing the model getting closer to the overall quadratic curve as the training run moves further along.

## Model Hyperparameters
* **Hidden Layers:** 4 fully connected layers.
* **Layer Width:** 128 nodes per hidden layer.
* **Activation:** LeakyReLU (\(alpha = 0.25\)).
* **Optimizer:** Adam.
* **Learning Rate:** Simulated through many runs to ensure fine tuning
* **Data Domain (Span):** Chosen by the user.

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
   Ensure you have `torch, numpy, and sklearn` installed:
   ```bash
   pip install torch
   pip install numpy
   pip install sklearn
   ```

---

## Usage

Run the script from your terminal to launch the interactive command-line interface:

```bash
python Polynomial_Regression.py
```

### CLI Options Available
* `[T]rain`: Trains the model for up to 50,000 epochs. Includes automated stopping and a nearly perfect learning rate.
* `[I]nference`: Test the trained model against true mathematical outputs by entering any number within the chosen span.
* `[Q]uit`: Exits the loop and terminates the program.

---

## Code Highlight: Data Generation
The code allows for comparisons of the model's progress across different time frames. It takes snippets from different points in the training, and compiles them into 4 different graphs that can be easily compared. It also generates a graph of the overall loss rate as a function of epochs.

### Results 
<img width="600" height="500" alt="model (3)" src="https://github.com/user-attachments/assets/2f95aa78-2c59-49b8-a437-b0bf945f64f2" />
<img width="1000" height="1000" alt="lossrate" src="https://github.com/user-attachments/assets/8c04ab6b-e915-4281-b115-5ef66f0f2f89" />


These are results of one run that I ran, showed on matplotlib. This was a run with a 0.01 error threshold, or 1%. The graph of the error shows that as time goes on, the model gets better and better. This is also reflected in the graph of the parabola, as the curve gets more and more accurate.

# What I Learned

In working on this project, I learned that learning models like this are incredible good at finding patterns in exponential functions. It was difficult to find where to start on this, but after I got a foothold in the basics, I just grew from there.

### I learned the following
1. How to use the Adam optimizer and refine the learning rate.
2. How to create an inference interface so the user can ask for an input-output.
3. Why more depth and layers of a model helps the model reach the correct answer faster.
4. I learned how to use matplotlib to model progress from the network.
5. I learned how matplotlib can be used to compare different architectures and results.
6. Scikit-learn is much faster and more accurate for simple problems like this.
7. How to add a model running multiple times with different learning rates to find the optimal one.

### Hardest Part
The hardest part was creating the scikit-learn model and the graph comparisons. This is simply because I don't use scikit-learn often.

## The Future
In the future, I'm probably going to add some normalization and move toward multi-variable. 


### Authors
* @maverickmhooper (https://www.github.com) --- Lead Developer

### License 
This project is licensed under the MIT License - see the LICENSE file for details.
Copyright © 2026 Maverick Hooper

