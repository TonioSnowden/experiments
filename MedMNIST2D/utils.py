import torch
import torch.nn as nn
from torchvision.models import resnet18
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import torch.nn.functional as F

class MultiTaskResNet18(nn.Module):
    def __init__(self, n_classes_mt, n_classes_ml):
        super(MultiTaskResNet18, self).__init__()
        self.resnet_base = resnet18(weights=None)
        
        in_features = self.resnet_base.fc.in_features
        
        self.resnet_base.fc = nn.Identity()
        
        self.fc_single = nn.Linear(in_features, n_classes_mt) 
        self.fc_multi = nn.Linear(in_features, n_classes_ml) 
    
    def forward(self, x, task_type):
        features = self.resnet_base(x)
        
        if task_type == "multi-label, binary-class":
            return self.fc_multi(features)
        elif task_type == 'multi-class':
            return self.fc_single(features)

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
    
class TextTargetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, label_dict, text_label, task, tokenizer):
        self.dataset = dataset
        self.label_dict = label_dict
        self.text_label = text_label
        self.task = task
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]

        if self.task == 'multi-label, binary-class':
            target_text = self.text_label + " with " + self.label_dict[str(target.item())]
        elif self.task == "multi-class":
            labels = [self.label_dict[str(i)] for i, label_present in enumerate(target) if label_present == 1]
            if len(labels) == 0:
                target_text = self.text_label + " with no issue to classify"
            else:
                target_text = self.text_label + " with " + ", ".join(labels)

        text_inputs = self.tokenizer(text=target_text, return_tensors="pt", padding='max_length', truncation=True, max_length=32)
        
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        
        return data, input_ids, attention_mask    
    
def info_nce_loss(image_features, text_features, temperature=0.07):
    """
    Calculates the InfoNCE loss between image and text features.
    """
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)

    similarity_matrix = torch.mm(image_features, text_features.t()) / temperature
    labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)

    loss_i = F.cross_entropy(similarity_matrix, labels)
    loss_t = F.cross_entropy(similarity_matrix.t(), labels)

    loss = (loss_i + loss_t) / 2
    return loss


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
    if task == 'multi-label, binary-class':
        return roc_auc_score(y_true, y_score)
    elif task == 'multi-class':
        return roc_auc_score(y_true, y_score, multi_class='ovr')


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