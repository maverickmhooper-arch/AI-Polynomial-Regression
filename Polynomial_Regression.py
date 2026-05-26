import torch
import torch.nn as nn



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

  print(f"Training to solve y = x\u00b2. Non-Linear Regression. ") # \u00b2 is the code for the exponent x^2

  print(f"Margin of error is {round((span**2)* 0.001, 2)}")

  success = False
  model.train()
  ema_loss = 0

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
        print(f"Epoch: {epoch: <3} | Loss: {loss.item():.2f}")

        if loss.item() <= (span**2) * 0.001:
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
    return model

  if not success:
    print("FAILURE IN TRAINING. ❌")
    print(f"EMA loss -> {ema_loss}")
    # PUNISHMENT
  return False # Signal to continue the main loop  

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
  


if __name__ == "__main__":
  x, y_target = vector(500)
  while True:
    user_input = input("[T]rain, [I]nference, [Q]uit. --> ").strip().upper()
    if user_input == "T":
      model = model_training()
    elif (user_input == "I"):
      if model:
        inference(model)
      else:
        print("No model exists. Train first. ")
  
    elif user_input == "Q":
      break  

  print("------END-------")
