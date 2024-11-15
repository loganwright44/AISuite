import os


from data.dataset_handler import ModelDataset
from torch.utils.data import DataLoader


train_dataset = ModelDataset(higgs_dataset_directory=os.path.dirname(os.path.abspath(__file__)) + '/datasets/data.parquet', training=True)
test_dataset = ModelDataset(higgs_dataset_directory=os.path.dirname(os.path.abspath(__file__)) + '/datasets/data.parquet', training=False)


IN_FEATURES = train_dataset.__getinfeatures__()
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 5


def data_loaders(train_batch_size: int = BATCH_SIZE_TRAIN, test_batch_size: int = BATCH_SIZE_TEST) -> dict:
  return {
    'train': DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True),
    'test': DataLoader(train_dataset, batch_size=test_batch_size, shuffle=False),
  }
