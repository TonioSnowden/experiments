import torch

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
    
class TargetOffsetDatasetCLIP(torch.utils.data.Dataset):
    def __init__(self, dataset, model, offset=0):
        self.dataset = dataset
        self.offset = offset
        self.model = model

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        data = self.model.get_image_features(data)
        adjusted_target = target + self.offset
        return data, adjusted_target
    
def calculate_num_classes(dataset):
    unique_classes = set()
    for _, target in dataset:
        unique_classes.add(target[0])
    return len(unique_classes)