from data_loader import load_data_training
from tqdm import tqdm
import torch
import torch.nn as nn
import clip
from transformers import CLIPTokenizer, CLIPModel

print("Load model")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

print("Load data")
train_loader, train_loader_at_eval, val_loader, test_loader = load_data_training(resize = True, download = True, as_rgb = True, tokenizer = clip_tokenizer, batch_size = 128)

device = "cuda:0" if torch.cuda.is_available() else "cpu" 
print("Fine-tune CLIP model")

clip_model = clip_model.to(device)
optimizer = torch.optim.Adam(clip_model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for fine tuning to new dataset
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    pbar = tqdm(train_loader, total=len(train_loader))
    for batch in pbar:
        optimizer.zero_grad()

        images,input_ids,attention_mask = batch 
        
        images= images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = clip_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

        total_loss.backward()
        
        optimizer.step()

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")
        
torch.save(clip_model.state_dict(), 'model.pth')