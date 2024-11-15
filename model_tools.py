import torch


def train(
    device: torch.device,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_dataloader: torch.utils.data.DataLoader,
    epochs: int = 50,
    loops: int = 5,
    verbose: bool = True
  ) -> None:
  
  # clarify types for vs-code
  x: torch.Tensor
  Y: torch.Tensor
  logits: torch.Tensor
  loss: torch.Tensor
  
  print("Beginning model training...")
  model.train()
  
  data_iterator = iter(training_dataloader)
  
  for loop in range(loops):
    epoch_avg_loss = 0.0
    
    for _ in range(epochs):
      x, Y = next(data_iterator)
      x, Y = x.to(device), Y.to(device)
      
      optimizer.zero_grad()
      
      logits, _ = model(x)
      
      loss = criterion(logits, Y)
      loss.backward()
      
      optimizer.step()
      
      epoch_avg_loss += loss.item()
    
    epoch_avg_loss /= epochs
    
    if verbose:
      print(f"Avg loss (loop {loop + 1}): {epoch_avg_loss: .4f}")
  
  print("Training finished!")
  return None


def evaluate(
    device: torch.device,
    model: torch.nn.Module,
    testing_dataloader: torch.utils.data.DataLoader,
    loops: int = 50
  ) -> None:
  
  # clarify types for vs-code
  x: torch.Tensor
  Y: torch.Tensor
  probabilities: torch.Tensor
  predictions: torch.Tensor
  total_correct: torch.Tensor = 0.0
  num_samples: int = 0
  
  print("Begining model evaluation...")
  model.eval()
  
  data_iterator = iter(testing_dataloader)
  
  with torch.no_grad():
    for _ in range(loops):
      x, Y = next(data_iterator)
      x, Y = x.to(device), Y.to(device)
      
      _, probabilities = model(x)
      
      predictions = (probabilities >= 0.5).int()
      
      total_correct += (predictions == Y.int()).sum().item()
      num_samples += Y.size(0)
  
  
  accuracy = total_correct / num_samples
  
  print()
  print(f"Model accuracy: {100 * accuracy: .4f} %")
  print()
  
  print("Evaluation finished!")
  return None


def save_weights(
    model: torch.nn.Module,
    filename: str = './model_weights/higgs.pt'
  ):
  try:
    torch.save(model.state_dict(), filename)
    print(f"Model weights saved to `{filename}`")
  except:
    print(f"Failed to save model weights to `{filename}`")



def load_weights(
    model: torch.nn.Module,
    filename: str = './model_weights/higgs.pt'
  ):
  try:
    model.load_state_dict(torch.load(filename))
    print(f"Model weights loaded from `{filename}`\n")
  except FileNotFoundError:
    print(f"File not found. Ignoring issue and returning...\n")

