import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dog_label_seq = {'안검종양_무': 0, '안검종양_유': 1, '안검내반증_무': 2, '색소침착성각막염_유': 3, '안검염_유': 4, '색소침착성각막염_무': 5, '백내장_무': 6, '핵경화_유': 7, '궤양성각막질환_상': 8, '안검내반증_유': 9, '백내장_비성숙': 10, '비궤양성각막질환_무': 11, 
                 '유루증_무': 12, '백내장_성숙': 13, '유루증_유': 14, '비궤양성각막질환_상': 15, '결막염_무': 16, '안검염_무': 17, '궤양성각막질환_하': 18, '핵경화_무': 19, '백내장_초기': 20, '궤양성각막질환_무': 21, '비궤양성각막질환_하': 22, '결막염_유': 23}

#cat_label_seq = {'안검염_유': 0, '비궤양성각막염_유': 1, '결막염_무': 2, '안검염_무': 3, '각막부골편_무': 4, '각막궤양_무': 5, '각막부골편_유': 6, '각막궤양_유': 7, '비궤양성각막염_무': 8, '결막염_유': 9}
cat_label_seq = {'결막염_무': 0, '결막염_유': 1}

class Pet_Dataset(Dataset):
    def __init__(self, df_path = "", transform_1 = None, transform_2 = None, cat_dog_label = ''):
        self.df = pd.read_csv(df_path)
        self.image_paths = self.df['image_path'].to_list()
        self.labels = self.df['label'].to_list()
        #self.transform_num = self.df['aug'].to_list()
        self.transform_num = 0
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.cat_dog_label = cat_dog_label
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        #img = cv2.imread(self.image_paths[index])
        img = Image.open(self.image_paths[index])
        
        label = self.labels[index]
        
        if self.cat_dog_label == 'cat':
            en_label = cat_label_seq[label]
        else:
            en_label = dog_label_seq[label]

        transform_num = 0

        if transform_num == 0:
            if self.transform_1:
                img = self.transform_1(img)
        
        if transform_num == 1:
            if self.transform_2:
                img = self.transform_2(img)
        
        return img, en_label

def pet_dataloader(df_path = "", transform_1 = None, transform_2 = None, batch_size = 8, shuffle = True, cat_dog_label = ""):
    dataset = Pet_Dataset(df_path = df_path, transform_1 = transform_1, transform_2 = transform_2, cat_dog_label = cat_dog_label)
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader

