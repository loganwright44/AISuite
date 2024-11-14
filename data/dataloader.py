import os


from data.dataset_handler import ModelDataset
from torch.utils.data import DataLoader


train_dataset = ModelDataset(higgs_dataset_directory=os.path.dirname(os.path.abspath(__file__)) + '/datasets/data.parquet', training=True)
test_dataset = ModelDataset(higgs_dataset_directory=os.path.dirname(os.path.abspath(__file__)) + '/datasets/data.parquet', training=False)


IN_FEATURES = train_dataset.__getinfeatures__()
BATCH_SIZE = 256


DataLoaders = {
  'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
  'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
}
