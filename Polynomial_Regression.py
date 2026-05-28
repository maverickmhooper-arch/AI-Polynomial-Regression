import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



def function(x):
  y = x**2
  return y

# Model Vectorization
def vector(span, num_generate=500):
  x = torch.linspace(-span, span, num_generate).view(-1, 1)
  y = function(x) # Removed noise for better results for training
  # Goal is a perfect parabola 
  return x, y # Return x and y


# 1. Model Definition

def scikit_model(x, y_target):
  x_values = x.numpy()
  y_values = y_target.numpy()
  poly_features = PolynomialFeatures(degree = 2, include_bias = False)
  x_ran = poly_features.fit_transform(x_values)
  poly_model = LinearRegression()
  poly_model.fit(x_ran, y_values)
  return poly_model, poly_features

def model_training(x, y_target, span, poly_model = None, poly_features = None, num_epochs = 50000):

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


  print(f"Margin of error is {round((function(span))* allowed_error, 10)}")

  scikit_loss = "N/A"
  if poly_model and poly_features :
    x_values = x.numpy()
    y_values = y_target.numpy()
    x_ran = poly_features.transform(x_values)
    scikit_prediction = poly_model.predict(x_ran)
    # Fixed: scikit_predictions -> scikit_prediction
    scikit_ran_error = np.mean((scikit_prediction - y_values) ** 2)
    scikit_loss = f"{scikit_ran_error}"

  success = False
  model.train()
  ema_loss = 0
  model_history = []
  losses = []
  epochs = []

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
    losses.append(loss.item())
    epochs.append(epoch)

    # Log Progress
    if (epoch % 100) == 0:
      with torch.no_grad():

        model.eval()
        current_prediction = model(x).numpy()
        model_history.append((epoch, current_prediction))
        model.train()
        print(f"Epoch: {epoch: <3} | Neural Loss: {loss.item():.8f}")


        if loss.item() <= (function(span)) * allowed_error:
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
    return model, model_history, losses, epochs

  if not success:
    print("FAILURE IN TRAINING. ❌")
    print(f"EMA loss -> {ema_loss}")
    # PUNISHMENT
  return False, [], losses, epochs # Signal to continue the main loop


def inference(model, span):

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
        input_val_np = np.array([[input_float]]) # Changed variable name to avoid shadowing
        # Fixed: .transfrom -> .transform
        # Fixed: Using the global poly_features and poly_model
        input_read = poly_features.transform(input_val_np)
        scikit_prediction = poly_model.predict(input_read)
        print(f"AI Prediction -> {round(prediction.item(), 2)}")
        print(f"Error was {abs(prediction.item() - (function(float(input_float))))}") # Use input_float
        # Fixed: Using [0,0] for scikit_prediction for robustness
        print(f"Scikit Prediction -> {round(scikit_prediction[0,0], 2)},"
)
        print(f"Error was {abs(scikit_prediction[0,0] - function(float(input_float)))}") # Use input_float
        print(f"Mathematical answer -> {function(float(input_float))}") # Use input_float

    except ValueError:
      print("Invalid input. Please enter a valid input. ")

def plot_history(x, y_target, history, poly_model = None, poly_features = None):
  history = history[1:]
  indices = [0, len(history)//3, 2*len(history)//3, len(history) - 1]

  scikit_line = None
  if poly_model and poly_features:
    x_ran = poly_features.transform(x.numpy())
    scikit_line = poly_model.predict(x_ran)

  fig, axes = plt.subplots(2, 2, figsize = (6, 5))
  axes = axes.flatten()
  for i, idx in enumerate(indices):
    if idx >= len(history):
      continue
    epoch, predictions = history[idx]
    ax = axes[i]

    # Fixed: .np() -> .numpy()
    ax.scatter(x.numpy(), y_target.numpy(), color = "blue", alpha = 0.3, s = 5, label = "Real")
    # Fixed: .np() -> .numpy()
    ax.plot(x.numpy(), predictions, color = "green", linewidth = 2, linestyle = "--", label = "Neural")

    if scikit_line is not None:
      ax.plot(x.numpy(), scikit_line, color = "red", linewidth = 1.5, linestyle = "-.", label = "Scikit Model")

    ax.set_title(f"Epoch {epoch}")
    ax.grid(True, linestyle = "--", alpha = 0.5)
    if i == 0:
      ax.legend()
  plt.title("Model vs Scikit vs Math")
  plt.tight_layout()
  plt.grid(True)
  plt.savefig("model.png")
  plt.show()
  time.sleep(10)
  plt.close()

def progress(epoch_list, loss_list):
  plt.figure(figsize = (10, 10))
  plt.plot(epoch_list, loss_list)
  plt.xlabel("Epoch")
  plt.ylabel("Error")
  plt.title("Error over time", fontsize = "20") # Changed plt.set_title to plt.title
  plt.yscale('log') # Set y-axis to logarithmic scale
  plt.savefig("lossrate.png")
  plt.show()
  time.sleep(10)
  plt.close()

def main():
  span = 50
  x, y_target = vector(span, 500)
  # Initialize these globally for inference function to access them
  global poly_model, poly_features
  poly_model = None
  poly_features = None

  while True:
    user_input = input("[T]rain, [I]nference, [Q]uit. --> ").strip().upper()
    if user_input == "T":
      # Assign to global variables
      poly_model, poly_features = scikit_model(x, y_target)
      model, history, losses, epochs = model_training(x, y_target, span, poly_model, poly_features)
      if model:
        plot_history(x, y_target, history, poly_model, poly_features)
        progress(epochs, losses)
    elif (user_input == "I"):
      if model and poly_model and poly_features: # Ensure all models are trained for inference
        inference(model, span)
      else:
        print("No models exist or are trained. Train first. ")

    elif user_input == "Q":
      break

  print("------END-------")




if __name__ == "__main__":
  main()
