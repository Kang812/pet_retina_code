import torch
import torchvision.transforms as transforms
import numpy as np

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

def score_function(real, pred):
    f1 = f1_score(real, pred, average="macro")
    acc = accuracy_score(real, pred)
    recall = recall_score(real, pred, average="macro")
    precision = precision_score(real, pred, average="macro")
    
    score_dict = dict()
    score_dict['f1_score'] = f1
    score_dict['accuracy'] = acc
    score_dict['recall'] = recall
    score_dict['precision'] = precision
    
    return score_dict

def class_per_accuracy(real_list, pred_list, label_seq):

    class_per_acc = dict()
    total_label = dict()
    encoder = dict()
    
    for it in label_seq.items():
        encoder[str(it[1])] = it[0]

    for i in range(len(encoder)):
        class_per_acc[encoder[str(i)]] = 0
        total_label[encoder[str(i)]] = 0

    for real, pred in zip(real_list, pred_list):
        if real == pred: 
            class_per_acc[encoder[str(real)]] +=1
        
        total_label[encoder[str(real)]] += 1
    
    return class_per_acc, total_label

def inference_results(model, dataframe, label_seq, transforms):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    range_int = dataframe.shape[0]
    
    real = []
    pred = []
    
    for i in tqdm(range(range_int)):
        img_path = dataframe.iloc[i]['image_path']
        
        real_class = label_seq[dataframe.iloc[i]['label']]
        real.append(real_class)
        
        img = Image.open(img_path)
        img = transforms(img)
        img = img.to(device)
        img = img.unsqueeze(0)

        with torch.no_grad():
            out = model(img)
            label_idx = torch.argmax(out, dim=1)
            label_idx = label_idx.detach().cpu().numpy().tolist()[0]
            pred.append(label_idx)
    
    results = score_function(real, pred)
    class_per_acc, total_label = class_per_accuracy(real, pred, label_seq)
    print()
    print('Inference Results:')
    print(' Accuracy:',np.round(results['accuracy'], 3),"%")
    print(' Recall:',np.round(results['recall'],3),"%")
    print(' Precision:',np.round(results['precision'],3),"%")
    print(' F1_score:',np.round(results['f1_score'],3),"%")
    print()
    print('Class 별 정확도:')
    for label in label_seq.keys():
        print(' Accuracy of %s: %d %%' % (label, 100 * class_per_acc[label] / total_label[label]))


    