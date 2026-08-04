import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys

# scripts 폴더 임포트 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KoJamoNet
from dataset import KoJamoDataset
from train_sft import sft_collate_fn, LengthGroupedSampler
from config import EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT

def profile():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    dataset = KoJamoDataset(data_dir="train_data_sft", seq_length=1000, stride=100, is_sft=True)
    sampler = LengthGroupedSampler(dataset, batch_size=2)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0, pin_memory=True, collate_fn=sft_collate_fn)
    vocab_sizes = dataset.tokenizer.get_vocab_sizes()
    
    model = KoJamoNet(vocab_sizes=vocab_sizes, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM,
                      num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
                      
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    
    print(f"Initial GPU memory: {torch.cuda.memory_allocated(device)/1e6:.2f} MB")
    
    model.train()
    accumulation_steps = 16
    optimizer.zero_grad()
    
    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        
        # Print shape and length of the batch
        print(f"\nStep {step} | Batch size: {x.size(0)} | Sequence Length: {x.size(1)}")
        print(f"GPU memory before forward: {torch.cuda.memory_allocated(device)/1e6:.2f} MB")
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits_type, logits_cho, logits_jung, logits_jong, logits_sym, logits_eng, logits_num = model(x, target_for_forcing=y)
            types_y = model._get_types(y)
            loss = criterion(logits_type.view(-1, 4), types_y.view(-1))
            
            loss_cho = criterion_none(logits_cho.view(-1, logits_cho.size(-1)), y[:, :, 0].view(-1))
            loss_jung = criterion_none(logits_jung.view(-1, logits_jung.size(-1)), y[:, :, 1].view(-1))
            loss_jong = criterion_none(logits_jong.view(-1, logits_jong.size(-1)), y[:, :, 2].view(-1))
            loss_sym = criterion_none(logits_sym.view(-1, logits_sym.size(-1)), y[:, :, 3].view(-1))
            loss_eng = criterion_none(logits_eng.view(-1, logits_eng.size(-1)), y[:, :, 4].view(-1))
            loss_num = criterion_none(logits_num.view(-1, logits_num.size(-1)), y[:, :, 5].view(-1))
            
            mask_korean = (types_y.view(-1) == 0).float()
            mask_sym = (types_y.view(-1) == 1).float()
            mask_eng = (types_y.view(-1) == 2).float()
            mask_num = (types_y.view(-1) == 3).float()
            
            loss += (loss_cho * mask_korean).sum() / (mask_korean.sum() + 1e-8)
            loss += (loss_jung * mask_korean).sum() / (mask_korean.sum() + 1e-8)
            loss += (loss_jong * mask_korean).sum() / (mask_korean.sum() + 1e-8)
            loss += (loss_sym * mask_sym).sum() / (mask_sym.sum() + 1e-8)
            loss += (loss_eng * mask_eng).sum() / (mask_eng.sum() + 1e-8)
            loss += (loss_num * mask_num).sum() / (mask_num.sum() + 1e-8)
            
            loss = loss / accumulation_steps
            
        print(f"GPU memory after forward: {torch.cuda.memory_allocated(device)/1e6:.2f} MB")
        
        loss.backward()
        print(f"GPU memory after backward: {torch.cuda.memory_allocated(device)/1e6:.2f} MB")
        
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"GPU memory after opt step: {torch.cuda.memory_allocated(device)/1e6:.2f} MB")
            
        if step > 40:
            break

if __name__ == "__main__":
    profile()
