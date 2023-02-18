import sys
sys.path.append("/mnt/d/project/pet_retina/utils/")
sys.path.append("/mnt/d/project/pet_retina/model/")
from train_model import train_classifer
from dataloader import pet_dataloader, Pet_Dataset
from load  import load_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import torch
from torch.optim import lr_scheduler

# train dataframe path
train_df_path = "/mnt/d/project/pet_retina/data/cat_train_conjunctivitis.csv"
val_df_path = "/mnt/d/project/pet_retina/data/cat_val_conjunctivitis.csv"
cat_dog_label = 'cat'

## transforms
"""train_transforms= A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(0.5),
    A.VerticalFlip(0.5),
    A.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ])

val_transforms= A.Compose([
    A.Resize(224, 224),
    A.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    ])"""

train_transforms_0 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_transforms_1 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

val_transforms= transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


## dataset load
dataset_sizes = dict()

train_dataset = Pet_Dataset(train_df_path, transform_1 = train_transforms_0, transform_2 = train_transforms_1, cat_dog_label = cat_dog_label)
val_dataset = Pet_Dataset(val_df_path, transform_1 = val_transforms, transform_2 = None, cat_dog_label = cat_dog_label)

dataset_sizes['train'] = len(train_dataset)
dataset_sizes['val'] = len(val_dataset)

## dataloader
data_loaders = dict()

shuffle = True
batch_size = 64
train_dataloader = pet_dataloader(df_path = train_df_path, transform_1 = train_transforms_0, transform_2 = train_transforms_1, batch_size = batch_size, shuffle = shuffle, cat_dog_label = cat_dog_label)
val_dataloader = pet_dataloader(df_path = val_df_path, transform_1 = val_transforms, transform_2 = None, batch_size = batch_size, shuffle = False, cat_dog_label = cat_dog_label)

data_loaders['train'] = train_dataloader
data_loaders['val'] = val_dataloader

## 모델 Load
model_name = 'efficientnet_b0'
pretrained = True
num_classes = 40

model = load_model(model_name, pretrained, num_classes)

## model training
epochs = 15
gpu_num = 1
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
device = torch.device("cuda:%s" % (gpu_num) if torch.cuda.is_available() else 'cpu')
output_path = "/mnt/d/project/pet_retina/work_dir/best.pt"
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_classifer(model, data_loaders, dataset_sizes, epochs, criterion, optimizer, scheduler, device, output_path)

