import sys
sys.path.append("./utils/")
sys.path.append("./model/")
from train_model import train_classifer
from dataloader import pet_dataloader, Pet_Dataset
from load  import load_model
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import torch
from torch.optim import lr_scheduler

# train dataframe path
# 결막염
##train_df_path = "/onetouch/project/pet_retina_develop/data/cat_train_conjunctivitis.csv"
##val_df_path = "/onetouch/project/pet_retina_develop/data/cat_val_conjunctivitis.csv"
##label_seq = {'결막염_무': 0, '결막염_유': 1}

# 각막궤양
##train_df_path = "/mnt/d/project/pet_retina_develop/data/cat_train_corneal_ulcer.csv"
##val_df_path = "/mnt/d/project/pet_retina_develop/data/cat_val_corneal_ulcer.csv"
##label_seq = {'각막궤양_무': 0, '각막궤양_유': 1}

# 각막부골편
#train_df_path = "/onetouch/project/pet_retina_develop/data/cat_train_feline_corneal_sequesration.csv"
#val_df_path = "/onetouch/project/pet_retina_develop/data/cat_val_feline_corneal_sequesration.csv"
#label_seq = {'각막부골편_무': 0, '각막부골편_유': 1}

# 안검염
##train_df_path = "/onetouch/project/pet_retina_develop/data/cat_train_meibomian_gland.csv"
##val_df_path = "/onetouch/project/pet_retina_develop/data/cat_val_meibomian_gland.csv"
##label_seq = {'안검염_무': 0, '안검염_유': 1}

# 비궤양성각막염
train_df_path = "/onetouch/project/pet_retina_develop/data/cat_train_non_ulcerative_corn_disease.csv"
val_df_path = "/onetouch/project/pet_retina_develop/data/cat_val_non_ulcerative_corn_disease.csv"
label_seq = {'비궤양성각막염_무': 0, '비궤양성각막염_유': 1}


train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_transforms= transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



## dataset load
dataset_sizes = dict()

train_dataset = Pet_Dataset(df_path = train_df_path, transform = train_transforms, label_seq = label_seq)
val_dataset = Pet_Dataset(df_path = val_df_path, transform = val_transforms, label_seq = label_seq)

dataset_sizes['train'] = len(train_dataset)
dataset_sizes['val'] = len(val_dataset)

## dataloader
data_loaders = dict()

shuffle = True
batch_size = 64
train_dataloader = pet_dataloader(df_path = train_df_path, transform = train_transforms, batch_size = batch_size, shuffle = True, label_seq = label_seq)
val_dataloader = pet_dataloader(df_path = val_df_path, transform = val_transforms, batch_size = 8, shuffle = True, label_seq = label_seq)

data_loaders['train'] = train_dataloader
data_loaders['val'] = val_dataloader

## 모델 Load
model_name = 'efficientnet_b0'
pretrained = True
num_classes = 2

model = load_model(model_name, pretrained, num_classes)

## model training
epochs = 15
gpu_num = 1
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
device = torch.device("cuda:%s" % (gpu_num) if torch.cuda.is_available() else 'cpu')
output_path = "/onetouch/project/pet_retina_develop/work_dir/best.pt"
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_classifer(model, data_loaders, dataset_sizes, epochs, criterion, optimizer, scheduler, device, output_path)

