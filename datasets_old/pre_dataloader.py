import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import glob
import random

class PreData(Dataset):
    def __init__(self,epoch):
        super().__init__()
        self.len = len
        self.list = glob.glob("./torch_data/{}/*.pth".format(epoch))
    
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        batch_data = torch.load(self.list[index])['data']
        return batch_data
    
    
def gDataloader(batch_size,epoch ,pin_memory = False ):
    dataset = PreData(epoch)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers = 6,pin_memory = pin_memory)
    return dataloader