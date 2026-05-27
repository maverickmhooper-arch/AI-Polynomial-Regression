import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as numpy
import time

# Model Vectorization
def vector(num_generate=500):
  global span
  span = 50
  x = torch.linspace(-span, span, num_generate).view(-1, 1)
  y = x**2 + torch.randn(x.size()) * 0.1 # Added noise with a magnitude of 0.1
  # Goal is a perfect parabola with minimal noise.
  return x, y # Return x and y
  

# 1. Model Definition

def model_training(num_epochs = 20000):

  model = nn.Sequential(
      nn.Linear(1, 128),
      nn.LeakyReLU(0.25),
      nn.Linear(128, 128),
      nn.LeakyReLU(0.25),
      nn.Linear(128, 128),
      nn.LeakyReLU(0.25),
      nn.Linear(128, 128),
      nn.LeakyReLU(0.25),
      nn.Linear(128, 1)
  )

  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
  criterion = nn.MSELoss()
  allowed_error = float(input("What would you like the allowed error to be? "))
  print(f"Training to solve y = x². Non-Linear Regression. ") # ² is the code for the exponent x^2


  print(f"Margin of error is {round((span**2)* allowed_error, 2)}")

  success = False
  model.train()
  ema_loss = 0
  model_history = []

  for epoch in range(num_epochs + 1):
    # Forward pass
    prediction = model(x)
    loss = criterion(prediction, y_target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch == 0:
      ema_loss = loss.item()

    alpha = 0.05
    ema_loss = (alpha * loss.item()) + (1 - alpha) * ema_loss

    # Log Progress
    if (epoch % 100) == 0:
      with torch.no_grad():
        model.eval()
        current_prediction = model(x).numpy()
        model_history.append((epoch, current_prediction))
        model.train()
        print(f"Epoch: {epoch: <3} | Loss: {loss.item():.2f}")

        if loss.item() <= (span**2) * allowed_error:
          print("SUCCESS IN TRAINING. ✅")
          print(f"EMA loss -> {ema_loss}")
          success = True
          break
      # GIVE REWARD


  # 4. Results
  print(f"Final Loss: {loss.item():.6f}")
  if success:
    print(f"Epochs required {epoch}.\n")
    print("AI finished training")
    return model, model_history

  if not success:
    print("FAILURE IN TRAINING. ❌")
    print(f"EMA loss -> {ema_loss}")
    # PUNISHMENT
  return False, [] # Signal to continue the main loop

def inference(model):

    try:
      input_val = input("What is the value that you would like to calculate? --> ")
      if input_val == "q":
        return True # Signal to quit the main loop
      input_float = float(input_val)
      if abs(input_float) > span:
        print("ERROR. INPUT VALUE OUT OF RANGE. ")
        return False
      y_inferred = torch.tensor([[input_float]])

      model.eval()
      with torch.no_grad():
        prediction = model(y_inferred)
        print(f"AI Prediction -> {round(prediction.item(), 2)}")
        print(f"Mathematical answer -> {float(input_val)**2}") # Cast input_val to float for calculation
        print(f"Error was {abs(prediction.item() - (float(input_val)**2))}") # Cast input_val to float for calculation
    except ValueError:
      print("Invalid input. Please enter a valid input. ")

def plot_history(x, y_target, history):
  history = history[1:]
  indices = [0, len(history)//3, 2*len(history)//3, len(history) - 1]

  fig, axes = plt.subplots(2, 2, figsize = (12, 10))
  axes = axes.flatten()
  for i, idx in enumerate(indices):
    if idx >= len(history):
      continue
    epoch, predictions = history[idx]
    ax = axes[i]

    ax.scatter(x.numpy(), y_target.numpy(), color = "red", alpha = 0.3, s = 5, label = "Real")
    ax.plot(x.numpy(), predictions, color = "green", linewidth = 2, label = "Model")

    ax.set_title(f"Epoch {epoch}")
    ax.grid(True, linestyle = "--", alpha = 0.5)
    if i == 0:
      ax.legend()
  plt.title("Model vs Math")
  plt.tight_layout()
  plt.grid(True)
  plt.savefig("model.png")
  plt.show()
  time.sleep(7.5)
  plt.close()


if __name__ == "__main__":
  x, y_target = vector(500)
  while True:
    user_input = input("[T]rain, [I]nference, [Q]uit. --> ").strip().upper()
    if user_input == "T":
      model, history = model_training()
      if model:
        plot_history(x, y_target, history)
    elif (user_input == "I"):
      if model:
        inference(model)
      else:
        print("No model exists. Train first. ")

    elif user_input == "Q":
      break

  print("------END-------")
