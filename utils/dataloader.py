import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Pet_Dataset(Dataset):
    def __init__(self, df_path = "", transform = None, label_seq = ""):
        self.df = pd.read_csv(df_path)
        self.image_paths = self.df['image_path'].to_list()
        self.labels = self.df['label'].to_list()
        self.transform = transform
        self.label_seq = label_seq
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        #img = cv2.imread(self.image_paths[index])
        img = Image.open(self.image_paths[index])
        
        label = self.labels[index]
        
        en_label = self.label_seq[label]
                
        if self.transform:
            img = self.transform(img)
        
        return img, en_label

def pet_dataloader(df_path = "", transform = None, batch_size = 8, shuffle = True, label_seq = ""):
    dataset = Pet_Dataset(df_path = df_path, transform = transform, label_seq = label_seq)
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader

