from tqdm import tqdm

import copy
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torch.utils.tensorboard import SummaryWriter

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

def train_classifer(model, dataloaders, dataset_sizes, epochs, criterion, optimizer, scheduler, device, output_path):

    model = model.to(device)    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0.0
    writer = SummaryWriter()

    for i in range(epochs):
        print('Epoch [%d/%d]' % (i + 1, epochs))
        print('-' * 10)

        train_pred = []
        train_y = []

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            pbar = tqdm(dataloaders[phase], unit='unit', unit_scale=True)

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                pbar.update(1)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                train_pred += preds.detach().cpu().numpy().tolist()
                train_y += labels.detach().cpu().numpy().tolist()
        
            if phase == 'train':
                scheduler.step()
        
            scores = score_function(train_y, train_pred)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = scores['accuracy']
            epoch_recall = scores['recall']
            epoch_f1score = scores['f1_score']
            epochs_precision = scores['precision']
            
            pbar.close()

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {epoch_recall:.4f} Precision: {epochs_precision:.4f} F1 Score: {epoch_f1score:.4f}')
            print()
            
            writer.add_scalar("Loss/%s" % (phase), epoch_loss, i + 1)
            writer.add_scalar("Accuracy/%s" % (phase), epoch_acc, i + 1)
            writer.add_scalar("Recall/%s" % (phase), epoch_recall, i + 1)
            writer.add_scalar("Precision/%s" % (phase), epochs_precision, i + 1)
            writer.add_scalar("F1score/%s" % (phase), epoch_f1score, i + 1)


            if phase == 'val' and epoch_f1score > best_score:
                best_score = epoch_f1score
                best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f'\nBest val f1 score: {best_score:4f}')
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), output_path)