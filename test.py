import sys
sys.path.append("./utils/")
sys.path.append("./model/")
import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms
import pandas as pd
from load import load_model
from inference import inference_results

val_transforms= transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_df = pd.read_csv("/onetouch/project/pet_retina_develop/data/cat_test_corneal_ulcer.csv")

label_seq = {'각막궤양_무': 0, '각막궤양_유': 1}

model_name = 'efficientnet_b0'
pretrained = False
num_classes = 2
model_path = '/onetouch/project/pet_retina_develop/work_dir/각막궤양/best.pt'

model = load_model(model_name, pretrained, num_classes)
model.load_state_dict(torch.load(model_path))

inference_results(model, test_df, label_seq, val_transforms)