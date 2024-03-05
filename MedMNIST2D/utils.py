import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


class MultiTaskResNet18(nn.Module):
    def __init__(self, num_classes_single, num_classes_multi):
        super(MultiTaskResNet18, self).__init__()
        # Initialize a ResNet18 model without pre-trained weights
        self.resnet_base = resnet18(weights=None)
        
        # Capture the number of input features to the final fully connected layer
        in_features = self.resnet_base.fc.in_features
        
        # Replace the original ResNet's final fully connected layer with Identity
        # This is now done after capturing in_features
        self.resnet_base.fc = nn.Identity()
        
        # Define new final layers for each task
        self.fc_single = nn.Linear(in_features, num_classes_single)  # For single-label classification
        self.fc_multi = nn.Linear(in_features, num_classes_multi) 
    
    def forward(self, x, task_type):
        # Extract features using the base ResNet model
        features = self.resnet_base(x)
        
        # Decide which task to perform based on the task_type argument
        if task_type == "multi-label, binary-class":
            return self.fc_single(features)
        elif task_type == 'multi-class':
            return self.fc_multi(features)

# Custom wrapper to adjust target labels
class TargetOffsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, offset=0):
        self.dataset = dataset
        self.offset = offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        adjusted_target = target + self.offset
        return data, adjusted_target

# Function to dynamically calculate the number of unique classes in a dataset
def calculate_num_classes(dataset):
    unique_classes = set()
    for _, target in dataset:
        unique_classes.add(target[0])
    return len(unique_classes)

def getAUC(y_true, y_score, task):
    """AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        ret = auc / y_score.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret


def getACC(y_true, y_score, task, threshold=0.5):
    """Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret
