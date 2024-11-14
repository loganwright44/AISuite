from torchsummary import summary
import torch
import os

"""
Outlines for each of the following custom modules have been laid
out for your convenience and to know how to use this tool. Modify
the modules below according to the model and data required by your
model. 
"""
from models.model import Model
from data.dataset_handler import ModelDataset
from argparsing import parse_args
from data.dataloader import (
  DataLoaders,
  IN_FEATURES
)
from model_tools import (
  train,
  evaluate,
  save_weights,
  load_weights
)

def main():
  SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
  DEVICE = torch.device('mps')
  
  # Change this to match the expected input of your model for the summary tool
  INPUT_SHAPE = (1, IN_FEATURES)
  ############################################################################

  CYCLES: int = 1
  TEST_LOOPS: int = 10
  TRAIN_LOOPS: int = 25
  EPOCHS: int = 100
  VERBOSE: bool = False
  COMPILE: bool
  
  args = parse_args()
  
  if args.cycles:
    CYCLES = args.cycles
  
  if args.testloops:
    TEST_LOOPS = args.testloops
  
  if args.trainloops:
    TRAIN_LOOPS = args.trainloops
  
  if args.epochs:
    EPOCHS = args.epochs
  
  VERBOSE = args.verbose
  COMPILE = args.compile
  
  model: torch.nn.Module = Model(in_features=IN_FEATURES, device=DEVICE)
  
  print()
  print(summary(model=model.to('cpu'), input_size=INPUT_SHAPE))
  print()
  
  model.to(device=DEVICE)
  
  if COMPILE:
    model = torch.compile(model, fullgraph=True, model='default')
  
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = torch.nn.BCEWithLogitsLoss()
  
  load_weights(model=model, filename=SCRIPT_DIR + '/model_weights/weights.pt')
  
  for cycle in range(CYCLES):
    print(f"Beginning cycle #{cycle + 1}:")
    
    train(
      device=DEVICE,
      model=model,
      criterion=criterion,
      optimizer=optimizer,
      training_dataloader=DataLoaders['train'],
      epochs=EPOCHS,
      loops=TRAIN_LOOPS,
      verbose=VERBOSE
    )
    evaluate(
      device=DEVICE,
      model=model,
      testing_dataloader=DataLoaders['test'],
      loops=TEST_LOOPS
    )
    
    print(f"Completed cycle #{cycle + 1}!")
  
  
  save_weights(model=model, filename=SCRIPT_DIR + '/model_weights/higgs_large.pt')
  
  
  
if __name__ == '__main__':
  main()


