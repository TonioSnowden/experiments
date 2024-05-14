import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import medmnist
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator
from models import ResNet18, ResNet50
from sklearn.metrics import roc_auc_score, accuracy_score
from tensorboardX import SummaryWriter
from torchvision.models import resnet18, resnet50
from tqdm import trange

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


def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, resize, as_rgb, model_path, run):

    lr = 0.001
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]
    datasets_2D = [
    ('pathmnist', 'PathMNIST'),
    #('chestmnist', 'ChestMNIST'),
    ('dermamnist', 'DermaMNIST'),
    ('octmnist', 'OCTMNIST'),
    ('pneumoniamnist', 'PneumoniaMNIST'),
    ('retinamnist', 'RetinaMNIST'),
    ('breastmnist', 'BreastMNIST'),
    ('bloodmnist', 'BloodMNIST'),
    ('tissuemnist', 'TissueMNIST'),
    ('organamnist', 'OrganAMNIST'),
    ('organcmnist', 'OrganCMNIST'),
    ('organsmnist', 'OrganSMNIST')
    ]

    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_ids[0])

    device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu') 
    
    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    if resize:
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    else:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])

# Concatenate datasets with adjusted target labels
    offset = 0
    concat_train_datasets = []
    concat_val_datasets = []
    concat_test_datasets = []

    for _, dataset in datasets_2D:
        DataClass = getattr(medmnist, dataset)
        temp_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
        num_classes = calculate_num_classes(temp_dataset)

        adjusted_train_dataset = TargetOffsetDataset(DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb), offset=offset)
        adjusted_val_dataset = TargetOffsetDataset(DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb), offset=offset)
        adjusted_test_dataset = TargetOffsetDataset(DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb), offset=offset)

        concat_train_datasets.append(adjusted_train_dataset)
        concat_val_datasets.append(adjusted_val_dataset)
        concat_test_datasets.append(adjusted_test_dataset)

        offset += num_classes

    train_dataset = ConcatDataset(concat_train_datasets)
    val_dataset = ConcatDataset(concat_val_datasets)
    test_dataset = ConcatDataset(concat_test_datasets)
        
    unique_classes = set()
    for _, target in train_dataset:
        unique_classes.add(target[0])
        
    task = "multi-class"
    n_channels = 3
    n_classes = len(unique_classes)

    # DataLoader setup
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print('==> Building and training model...')
    
    if model_flag == 'resnet18':
        model =  resnet18(pretrained=False, num_classes=n_classes) if resize else ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_flag == 'resnet50':
        model =  resnet50(pretrained=False, num_classes=n_classes) if resize else ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    model = model.to(device)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device)['net'], strict=True)
        train_metrics = test(model, train_loader_at_eval, task, criterion, device, run, output_root)
        val_metrics = test(model, val_loader, task, criterion, device, run, output_root)
        test_metrics = test(model, test_loader, task, criterion, device, run, output_root)

        print('train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2]) + \
              'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2]) + \
              'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2]))

    if num_epochs == 0:
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_'+log for log in logs]
    val_logs = ['val_'+log for log in logs]
    test_logs = ['test_'+log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs+val_logs+test_logs, 0)
    
    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)

    global iteration
    iteration = 0
    
    for epoch in trange(num_epochs):        
        train_loss = train(model, train_loader, task, criterion, optimizer, device, writer)
        
        train_metrics = test(model, train_loader_at_eval, task, criterion, device, run)
        val_metrics = test(model, val_loader, task, criterion, device, run)
        test_metrics = test(model, test_loader, task, criterion, device, run)
        
        scheduler.step()
        
        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        for key, value in log_dict.items():
            writer.add_scalar(key, value, epoch)
            
        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = {
        'net': best_model.state_dict(),
    }

    path = os.path.join(output_root, 'best_model.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_loader_at_eval, task, criterion, device, run, output_root)
    val_metrics = test(best_model, val_loader, task, criterion, device, run, output_root)
    test_metrics = test(best_model, test_loader, task, criterion, device, run, output_root)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' % (data_flag) + train_log + val_log + test_log
    print(log)
            
    with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
        f.write(log)  
            
    writer.close()


def train(model, train_loader, task, criterion, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1

        loss.backward()
        optimizer.step()
    
    epoch_loss = sum(total_loss)/len(total_loss)
    return epoch_loss


def test(model, data_loader, task, criterion, device, run, save_folder=None):
    model.eval()
    
    total_loss = []
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if len(targets) == 14:
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs)
            else:
                targets = torch.squeeze(targets, 1).long()
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs)

            total_loss.append(loss.item())
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    # Convert lists to Numpy arrays
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    # Calculate AUC and ACC
    if task == 'multi-label, binary-class' or task == 'binary-class':
        # For binary classification and multi-label (consider each label separately)
        try:
            auc = roc_auc_score(all_targets, all_predictions, multi_class='ovr')
        except ValueError:
            # Handle cases where only one class is present in `y_true` or other issues
            auc = float('nan')
    else:
        # Assuming multi-class classification
        # Convert predictions to class values
        predictions_class = np.argmax(all_predictions, axis=1)
        auc = roc_auc_score(all_targets, all_predictions, multi_class='ovr')
        acc = accuracy_score(all_targets, predictions_class)

    test_loss = np.mean(total_loss)

    return [test_loss, auc, acc]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST2D')

    parser.add_argument('--data_flag',
                        default='pathmnist',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epochs',
                        default=3,
                        help='num of epochs of training, the script would only test model if set num_epochs to 0',
                        type=int)
    parser.add_argument('--gpu_ids',
                        default='0',
                        type=str)
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize',
                        help='resize images of size 28x28 to 224x224',
                        action="store_true")
    parser.add_argument('--as_rgb',
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_path',
                        default=None,
                        help='root of the pretrained model to test',
                        type=str)
    parser.add_argument('--model_flag',
                        default='resnet18',
                        help='choose backbone from resnet18, resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)


    args = parser.parse_args()
    data_flag = args.data_flag
    output_root = args.output_root
    num_epochs = args.num_epochs
    gpu_ids = args.gpu_ids
    batch_size = args.batch_size
    download = True
    model_flag = args.model_flag
    resize = args.resize
    as_rgb = True
    model_path = args.model_path
    run = args.run
    
    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, resize, as_rgb, model_path, run)