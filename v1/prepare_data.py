import json
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import decoders
from datasets import load_dataset
import os
from tqdm import tqdm

def prepare_tokenizer_and_data(dpo_path="dpo_5000_samples.jsonl", vocab_size=32000, save_path="custom_tokenizer.json", max_articles=50000, max_seq_len=512):
    print("🚀 [1/4] 위키백과 데이터 다운로드 중...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train", streaming=True)
    
    texts = []
    articles = []
    count = 0
    progress = tqdm(total=max_articles, desc="문서 다운로드")
    for item in dataset:
        texts.append(item['text'])
        articles.append(item['text'])
        count += 1
        progress.update(1)
        if count >= max_articles:
            break
    progress.close()
            
    print("🚀 [2/4] DPO 추론 데이터 로딩 중...")
    if os.path.exists(dpo_path):
        with open(dpo_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text_chosen = f"[문제]\n{data['prompt']}\n\n[정답 풀이 (Chosen)]\n{data['chosen']}<eos>"
                text_rejected = f"[문제]\n{data['prompt']}\n\n[오답 풀이 (Rejected)]\n{data['rejected']}<eos>"
                texts.extend([text_chosen, text_rejected])
    else:
        print(f"⚠️ {dpo_path} 파일이 없습니다. 위키백과로만 토크나이저를 학습합니다.")

    print(f"🚀 [3/4] BPE 토크나이저 학습 시작 (단어 수: {vocab_size})... 이 작업은 몇 분 정도 걸립니다.")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<pad>", "<eos>", "<bos>"])
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(save_path)
    print(f"✅ 토크나이저가 완벽하게 학습되어 {save_path}에 저장되었습니다!")

    print(f"🚀 [4/4] 학습 속도 향상을 위해 위키백과를 청크(Chunk)로 쪼개어 저장합니다...")
    chunks = []
    progress_bar = tqdm(total=len(articles), desc="문서 토큰화 및 저장")
    for text in articles:
        enc = tokenizer.encode(text + "<eos>")
        ids = enc.ids
        for j in range(0, len(ids), max_seq_len):
            chunk = ids[j:j+max_seq_len]
            if len(chunk) > 10:
                chunks.append(chunk)
        progress_bar.update(1)
    progress_bar.close()
    
    torch.save(chunks, "wiki_chunks.pt")
    print(f"✅ 전처리 완벽 종료! 총 {len(chunks)}개의 청크가 'wiki_chunks.pt' 파일로 저장되었습니다.")
    print("👉 이제 언제든지 'python train.py --mode pretrain_wiki'를 즉시 실행할 수 있습니다!")

if __name__ == "__main__":
    prepare_tokenizer_and_data()
