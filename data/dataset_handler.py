import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple


class ModelDataset(Dataset):
  def __init__(self, dataset_directory = './datasets/data.parquet', training: bool = False):
    self.dir = dataset_directory
    self.training = training
    
    print(f"{'Train' if training else 'Test'} `ModelDataset` init:")
    print(f"Reading `{dataset_directory}`, may take a moment....")
    
    self.data = pd.read_parquet(dataset_directory, engine='pyarrow')
    
    print(f"Data preparation complete!\n")
    
    # columns do not need to be set because of the dataframe having been saved to a .parquet file type
    #self.data.columns = [
    #    "signal", "lepton pT", "lepton eta", "lepton phi",
    #    "missing energy magnitude", "missing energy phi",
    #    "jet 1 pt", "jet 1 eta", "jet 1 phi", "jet 1 b-tag",
    #    "jet 2 pt", "jet 2 eta", "jet 2 phi", "jet 2 b-tag",
    #    "jet 3 pt", "jet 3 eta", "jet 3 phi", "jet 3 b-tag",
    #    "jet 4 pt", "jet 4 eta", "jet 4 phi", "jet 4 b-tag",
    #    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
    #]
    
    if training:
      self.labels = torch.tensor(self.data.iloc[:-500_000, 0].values, dtype=torch.float32)
      self.tensor_data = torch.tensor(self.data.iloc[:-500_000, 1:].values, dtype=torch.float32)
      self.len = len(self.labels)
      self._in_features = len(self.tensor_data[0])
      del self.data
    else:
      self.labels = torch.tensor(self.data.iloc[-500_000:, 0].values, dtype=torch.float32)
      self.tensor_data = torch.tensor(self.data.iloc[-500_000:, 1:].values, dtype=torch.float32)
      self.len = len(self.labels)
      self._in_features = len(self.tensor_data[0])
      del self.data
  
  
  def __len__(self) -> int:
    return self.len
  
  
  def __getinfeatures__(self) -> int:
    return self._in_features
  
  
  def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
    """ __getitem__ returns a Tuple of Tensors

    Args:
        index (int): idx of dataset

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: data, label
    """
    return self.tensor_data[index, :], self.labels[index].unsqueeze(dim=0)


