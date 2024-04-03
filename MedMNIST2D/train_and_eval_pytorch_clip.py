import argparse
import os
import time
import medmnist
from medmnist import INFO
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from tqdm import trange
from transformers import CLIPTokenizer, CLIPModel
import PIL

# Import utilities from utils.py
from utils import TextTargetDataset

def main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, resize, as_rgb, model_path, run):

    lr = 0.001
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]
    datasets_2D = [
    ('pathmnist', 'PathMNIST', 'Histological images of colorectal cancer tissue patches'),
    #('chestmnist', 'ChestMNIST'),
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
    multi_label_datasets_2D = [
        ('chestmnist', 'ChestMNIST', 'Frontal-view X-ray images of chest scans')]
        
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
    print(output_root)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
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

    concat_train_datasets = []
    concat_val_datasets = []
    concat_test_datasets = []
    
    #Multi Label loading
    task = 'multi-label, binary-class'

    for data_flag, dataset, text_label in datasets_2D:
        info = INFO[data_flag]
        DataClass = getattr(medmnist, dataset)
        label_dict = info['label']

        train_dataset = TextTargetDataset(DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb), label_dict, text_label, task, clip_tokenizer)
        val_dataset = TextTargetDataset(DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb), label_dict, text_label, task, clip_tokenizer)
        test_dataset = TextTargetDataset(DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb), label_dict, text_label, task, clip_tokenizer)

        concat_train_datasets.append(train_dataset)
        concat_val_datasets.append(val_dataset)
        concat_test_datasets.append(test_dataset)

    mc_train_dataset = ConcatDataset(concat_train_datasets)
    mc_val_dataset = ConcatDataset(concat_val_datasets)
    mc_test_dataset = ConcatDataset(concat_test_datasets)
        
    #Multi Class loading
    task = "multi-class"
    
    #Load ChestMNIST dataset
    data_flag = multi_label_datasets_2D[0][0]
    dataset = multi_label_datasets_2D[0][1]
    text_label = multi_label_datasets_2D[0][2]
    
    info = INFO[data_flag]
    DataClass = getattr(medmnist, multi_label_datasets_2D[0][1])
    label_dict = info['label']
    
    ml_train_dataset = TextTargetDataset(DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb), label_dict, text_label, task, clip_tokenizer)
    ml_test_dataset = TextTargetDataset(DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb), label_dict, text_label, task, clip_tokenizer)
    ml_val_dataset = TextTargetDataset(DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb), label_dict, text_label, task, clip_tokenizer)
    
    train_dataset = ConcatDataset([mc_train_dataset, ml_train_dataset])
    val_dataset = ConcatDataset([mc_val_dataset, ml_val_dataset])
    test_dataset = ConcatDataset([mc_test_dataset, ml_test_dataset])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    model = clip_model
    model = model.to(device)

    if num_epochs == 0:
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    best_loss = float('inf')
    best_epoch = 0
    best_model = None
    
    global iteration
    iteration = 0

    writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

    for epoch in trange(num_epochs):        
        train_loss = train(model, train_loader, optimizer, device, writer)
                
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            best_model = model.state_dict()
        
        print(train_loss)
        
    print('==> Finished training...')
    print('Best Loss: {:.3f} at epoch {}'.format(train_loss, best_epoch))
    #save model
    print("saved at", output_root)
    torch.save(best_model, os.path.join(output_root, 'best_model.pth'))
        
def train(model, train_loader, optimizer, device, writer):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (images, input_ids, attention_mask) in enumerate(train_loader):
        optimizer.zero_grad()
        
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images, return_loss=True)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        writer.add_scalar('train_loss_logs', loss.item(), iteration)
        iteration += 1
    
    epoch_loss = sum(total_loss) / len(total_loss)
    return epoch_loss

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
                        default=5,
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
    resize = True
    as_rgb = True
    model_path = args.model_path
    run = args.run
    
    main(data_flag, output_root, num_epochs, gpu_ids, batch_size, download, model_flag, resize, as_rgb, model_path, run)