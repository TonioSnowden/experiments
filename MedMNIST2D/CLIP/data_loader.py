import argparse
import os
import time
import medmnist
from medmnist import INFO
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as transforms
import PIL
from utils import *

MULTI_CLASS_DATASETS = [
    ('pathmnist', 'PathMNIST', 'Histological images of colorectal cancer tissue patches'),
    ('dermamnist', 'DermaMNIST','Dermatoscopic images of common pigmented skin lesions'),
    ('octmnist', 'OCTMNIST', 'Optical coherence tomography images for retinal diseases'),
    ('pneumoniamnist', 'PneumoniaMNIST' , 'Pediatric chest X-ray images'),
    ('retinamnist', 'RetinaMNIST', 'Retina fundus images'),
    ('breastmnist', 'BreastMNIST', 'Breast ultrasound images'),
    ('bloodmnist', 'BloodMNIST', 'Blood cell images'),
    ('tissuemnist', 'TissueMNIST', 'Human kidney cortex cells'),
    ('organamnist', 'OrganAMNIST', 'axial views of 3D computed tomography (CT) images of 11 body organs'),
    ('organcmnist', 'OrganCMNIST', 'coronal views of 3D computed tomography (CT) images of 11 body organs'),
    ('organsmnist', 'OrganSMNIST', 'sagittal views of 3D computed tomography (CT) images of 11 body organs')
]
MULTI_LABEL_DATASETS = [
    ('chestmnist', 'ChestMNIST', 'Frontal-view X-ray images of chest scans')]

def load_data_training(resize, download, as_rgb, tokenizer, batch_size):
    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
            transforms.Grayscale(num_output_channels=3),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])]) 
    else:
        data_transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])]) 
        
    task = 'multi-label, binary-class'
    
    concat_train_datasets = []
    concat_val_datasets = []
    concat_test_datasets = []
    
    print("Create dataset")

    for data_flag, dataset, text_label in MULTI_CLASS_DATASETS:
        info = INFO[data_flag]
        DataClass = getattr(medmnist, dataset)
        label_dict = info['label']

        train_dataset = TextTargetDataset(DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb), label_dict, text_label, task, tokenizer)
        val_dataset = TextTargetDataset(DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb), label_dict, text_label, task, tokenizer)
        test_dataset = TextTargetDataset(DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb), label_dict, text_label, task, tokenizer)

        concat_train_datasets.append(train_dataset)
        concat_val_datasets.append(val_dataset)
        concat_test_datasets.append(test_dataset)
        
    mc_train_dataset = ConcatDataset(concat_train_datasets)
    mc_val_dataset = ConcatDataset(concat_val_datasets)
    mc_test_dataset = ConcatDataset(concat_test_datasets)
        
    #Multi Class loading
    task = "multi-class"
    
    #Load ChestMNIST dataset
    data_flag = MULTI_LABEL_DATASETS[0][0]
    dataset = MULTI_LABEL_DATASETS[0][1]
    text_label = MULTI_LABEL_DATASETS[0][2]
    
    info = INFO[data_flag]
    DataClass = getattr(medmnist, MULTI_LABEL_DATASETS[0][1])
    label_dict = info['label']
    
    ml_train_dataset = TextTargetDataset(DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb), label_dict, text_label, task, tokenizer)
    ml_test_dataset = TextTargetDataset(DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb), model, tokenizer)
    ml_val_dataset = TextTargetDataset(DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb), label_dict, text_label, task, tokenizer)
    
    train_dataset = ConcatDataset([mc_train_dataset, ml_train_dataset])
    val_dataset = ConcatDataset([mc_val_dataset, ml_val_dataset])
    test_dataset = ConcatDataset([mc_test_dataset, ml_test_dataset])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, train_loader_at_eval, val_loader, test_loader
        
        
        
        
        
        
        
def load_data_clip(resize, download, as_rgb, model, batch_size):
    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
            
    offset = 0
    concat_train_datasets = []
    concat_val_datasets = []
    concat_test_datasets = []
    
    print("Create dataset")

    for _, dataset, _ in MULTI_CLASS_DATASETS:
        DataClass = getattr(medmnist, dataset)
        temp_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)        
        num_classes = calculate_num_classes(temp_dataset)

        train_dataset = TargetOffsetDatasetCLIP(DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb),  model, offset = offset)
        val_dataset = TargetOffsetDatasetCLIP(DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb),  modeloffset = offset)
        test_dataset = TargetOffsetDatasetCLIP(DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb),  modeloffset = offset)

        concat_train_datasets.append(train_dataset)
        concat_val_datasets.append(val_dataset)
        concat_test_datasets.append(test_dataset)
        
        offset += num_classes

        
    train_dataset = ConcatDataset(concat_train_datasets)
    val_dataset = ConcatDataset(concat_val_datasets)
    test_dataset = ConcatDataset(concat_test_datasets)  
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, train_loader_at_eval, val_loader, test_loader, train_dataset
    
    

