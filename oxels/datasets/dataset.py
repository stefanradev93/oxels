from collections.abc import Sequence, Callable

from torch.utils.data import Dataset


class OxelDataset(Dataset):
    def __init__(self, files: Sequence[str], augmentation: Callable):
        
        self.files = files
        self.augmenation = augmentation
        
        # TODO

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):      
        
        sub_inputs, sub_outputs, masks = None, None, None
        
        sub_inputs = [self.augmentation(im) for im in sub_inputs]
            
        return input, output, masks
