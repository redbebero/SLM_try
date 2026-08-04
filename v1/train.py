import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import MicroHRMDeepSeek
from tqdm import tqdm
import os
from datasets import load_dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()

class WikiPretrainDataset(Dataset):
    def __init__(self, tokenizer_path, max_seq_len=512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        
        data_path = "wiki_chunks.pt"
        print(f"📚 미리 전처리된 위키백과 데이터({data_path})를 즉시 로딩합니다...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"🚨 {data_path} 파일이 없습니다! 먼저 'python prepare_data.py'를 실행하여 데이터를 생성해주세요.")
            
        self.chunks = torch.load(data_path)
        print(f"✅ 위키백과 청크(Chunk) 준비 완료: 총 {len(self.chunks)}개 배정")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def collate_fn(self, batch):
        x_list, y_list = zip(*batch)
        raw_max_len = max([x.size(0) for x in x_list])
        max_len = math.ceil(raw_max_len / 64) * 64 if raw_max_len < self.max_seq_len else self.max_seq_len
        
        pad_x = torch.stack([F.pad(x, (0, max_len - x.size(0)), value=self.pad_id) for x in x_list])
        pad_y = torch.stack([F.pad(y, (0, max_len - y.size(0)), value=-100) for y in y_list])
        return pad_x, pad_y, None, None # Match DPO format superficially for the train loop

class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer_path, max_seq_len=512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        
        prompt = f"[문제]\n{data['prompt']}\n\n"
        chosen_ans = f"[정답 풀이 (Chosen)]\n{data['chosen']}<eos>"
        rejected_ans = f"[오답 풀이 (Rejected)]\n{data['rejected']}<eos>"
        
        p_enc = self.tokenizer.encode(prompt)
        c_enc = self.tokenizer.encode(chosen_ans)
        r_enc = self.tokenizer.encode(rejected_ans)
        
        chosen_ids = (p_enc.ids + c_enc.ids)[:self.max_seq_len]
        c_x = chosen_ids[:-1]
        c_label = ([-100] * len(p_enc.ids)) + c_enc.ids
        c_label = c_label[1:self.max_seq_len]
        
        rejected_ids = (p_enc.ids + r_enc.ids)[:self.max_seq_len]
        r_x = rejected_ids[:-1]
        r_label = ([-100] * len(p_enc.ids)) + r_enc.ids
        r_label = r_label[1:self.max_seq_len]
        
        return (
            torch.tensor(c_x, dtype=torch.long), torch.tensor(c_label, dtype=torch.long),
            torch.tensor(r_x, dtype=torch.long), torch.tensor(r_label, dtype=torch.long)
        )

    def collate_fn(self, batch):
        c_x_list, c_y_list, r_x_list, r_y_list = zip(*batch)
        
        raw_max_len_c = max([x.size(0) for x in c_x_list])
        raw_max_len_r = max([x.size(0) for x in r_x_list])
        
        max_len_c = math.ceil(raw_max_len_c / 64) * 64 if raw_max_len_c < self.max_seq_len else self.max_seq_len
        max_len_r = math.ceil(raw_max_len_r / 64) * 64 if raw_max_len_r < self.max_seq_len else self.max_seq_len
        
        pad_c_x = torch.stack([F.pad(x, (0, max_len_c - x.size(0)), value=self.pad_id) for x in c_x_list])
        pad_c_y = torch.stack([F.pad(y, (0, max_len_c - y.size(0)), value=-100) for y in c_y_list])
        
        pad_r_x = torch.stack([F.pad(x, (0, max_len_r - x.size(0)), value=self.pad_id) for x in r_x_list])
        pad_r_y = torch.stack([F.pad(y, (0, max_len_r - y.size(0)), value=-100) for y in r_y_list])
        
        return pad_c_x, pad_c_y, pad_r_x, pad_r_y

def train(data_path="dpo_5000_samples.jsonl", mode="pretrain_wiki"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device} | 🚀 Mode: {mode.upper()}")
    
    tokenizer_path = "custom_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print("🚨 오류: custom_tokenizer.json 파일이 없습니다. 먼저 'python prepare_data.py'를 실행해주세요!")
        return

    # 1. 데이터 로드
    if mode == "pretrain_wiki":
        dataset = WikiPretrainDataset(tokenizer_path, max_seq_len=512)
    elif mode == "dpo":
        dataset = DPODataset(data_path, tokenizer_path, max_seq_len=512)
    else:
        raise ValueError("Invalid mode. Use 'pretrain_wiki' or 'dpo'")
    
    from torch.utils.data import random_split
    val_size = max(1, int(len(dataset) * 0.05)) # 5% validation
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    batch_size = 4 
    accumulation_steps = 16
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=2, pin_memory=True)
    
    # 토크나이저에서 Vocab 사이즈 가져오기 (예: 32000)
    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    print(f"Vocab Size: {vocab_size}")

    model = MicroHRMDeepSeek(
        vocab_size=vocab_size, 
        dim=768, 
        num_heads=12, 
        latent_dim=64, 
        mlp_hidden_dim=3072, 
        max_seq_len=512, 
        thinking_steps=5
    ).to(device)
    
    import copy
    if mode == "dpo":
        ref_model = copy.deepcopy(model).to(device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model = torch.compile(ref_model, dynamic=True)
    else:
        ref_model = None
    
    print("컴파일 중... (PyTorch 2.0+)")
    model = torch.compile(model, dynamic=True)
    
    learning_rate = 5e-4 if mode == "pretrain_wiki" else 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    sft_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    epochs = 100 if mode == "pretrain_wiki" else 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    os.makedirs("checkpoints", exist_ok=True)
    import glob
    checkpoints = glob.glob("checkpoints/micro_hrm_deepseek_ep*.pt")
    start_epoch = 0
    if checkpoints:
        latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_ep')[1].split('.pt')[0]))
        checkpoint_data = torch.load(latest_ckpt, map_location=device)
        
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            # 새로운 포맷 (모든 상태 저장)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
            if mode == "dpo" and ref_model is not None:
                ref_model.load_state_dict(checkpoint_data['model_state_dict'])
            start_epoch = int(latest_ckpt.split('_ep')[1].split('.pt')[0])
            print(f"✅ 완벽한 체크포인트 로드 성공 (Optimizer 유지): {latest_ckpt}")
        else:
            # 과거 포맷 (모델 가중치만 존재)
            model.load_state_dict(checkpoint_data)
            if mode == "dpo" and ref_model is not None:
                ref_model.load_state_dict(checkpoint_data)
            start_epoch = int(latest_ckpt.split('_ep')[1].split('.pt')[0])
            # 옵티마이저 정보가 없으므로 스케줄러 강제 억지로 전진
            for _ in range(start_epoch):
                optimizer.step()
                scheduler.step()
            print(f"⚠️ 구형 체크포인트 로드 (모델 가중치만 복구됨): {latest_ckpt}")
        
    print(f"🏃 학습 시작! 총 {epochs} Epochs... (Train: {train_size}, Val: {val_size})")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, (c_x, c_y, r_x, r_y) in enumerate(progress_bar):
            c_x, c_y = c_x.to(device, non_blocking=True), c_y.to(device, non_blocking=True)
            if mode == "dpo":
                r_x, r_y = r_x.to(device, non_blocking=True), r_y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                policy_chosen_logps, chosen_logits = model.compute_log_probs(c_x, c_y, return_logits=True)
                sft_loss_val = sft_loss_fn(chosen_logits.view(-1, chosen_logits.size(-1)), c_y.view(-1))
                
                if mode == "dpo":
                    policy_rejected_logps, _ = model.compute_log_probs(r_x, r_y, return_logits=True)
                    with torch.no_grad():
                        ref_chosen_logps, _ = ref_model.compute_log_probs(c_x, c_y, return_logits=True)
                        ref_rejected_logps, _ = ref_model.compute_log_probs(r_x, r_y, return_logits=True)
                    
                    dpo_loss_val = compute_dpo_loss(
                        policy_chosen_logps, policy_rejected_logps,
                        ref_chosen_logps, ref_rejected_logps, beta=0.1
                    )
                    loss = sft_loss_val + (0.5 * dpo_loss_val)
                else:
                    loss = sft_loss_val
                
                step_loss = loss / accumulation_steps
            
            scaler.scale(step_loss).backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            total_loss += loss.detach().item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for c_x, c_y, r_x, r_y in val_loader:
                c_x, c_y = c_x.to(device, non_blocking=True), c_y.to(device, non_blocking=True)
                if mode == "dpo":
                    r_x, r_y = r_x.to(device, non_blocking=True), r_y.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    _, chosen_logits = model.compute_log_probs(c_x, c_y, return_logits=True)
                    v_loss = sft_loss_fn(chosen_logits.view(-1, chosen_logits.size(-1)), c_y.view(-1))
                    val_loss += v_loss.item()
                    
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        scheduler.step()
        
        # 모델의 뇌 상태, 모멘텀, 학습률 등을 통째로 저장
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }
        torch.save(checkpoint, f"checkpoints/micro_hrm_deepseek_ep{epoch+1}.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MicroHRMDeepSeek 학습 스크립트")
    parser.add_argument("--data_path", type=str, default="dpo_5000_samples.jsonl", help="DPO 학습용 데이터셋 경로")
    parser.add_argument("--mode", type=str, choices=["pretrain_wiki", "dpo"], default="pretrain_wiki", help="학습 모드: pretrain_wiki (위키백과 기초학습) or dpo (추론 학습)")
    args = parser.parse_args()
    train(data_path=args.data_path, mode=args.mode)
