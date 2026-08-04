import torch
from torch.utils.data import Dataset, DataLoader
import os
from tokenizer import KoJamoTokenizer


class KoJamoDataset(Dataset):
    """
    텍스트 파일을 읽어서 (Seq_len, 6) 텐서로 변환하여
    Next-token 예측을 위한 (input, target) 텐서 쌍을 반환합니다.
    """
    def __init__(self, data_dir="train_data", seq_length=50, stride=10):
        self.tokenizer  = KoJamoTokenizer()
        self.seq_length = seq_length
        self.stride     = stride  # 속도 향상을 위한 건너뛰기 보폭

        # train_data 폴더에서 모든 txt 수집
        import glob
        full_text = ""
        
        if os.path.isdir(data_dir):
            txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
            if txt_files:
                print(f"📖 데이터 폴더에서 텍스트 수집 완료: {len(txt_files)}개 파일 로드 중...")
                texts = []
                for fp in sorted(txt_files):
                    with open(fp, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                full_text = "\n\n".join(texts)
            
        # 만약 폴더에 텍스트가 없거나 경로가 폴더가 아닐 때의 예외 대체
        if not full_text:
            fallback = "datasets/wiki_clean_short.txt"
            if os.path.exists(fallback):
                print(f"⚠️ 폴더에 데이터가 없어 {fallback} 파일을 사용합니다.")
                with open(fallback, "r", encoding="utf-8") as f:
                    full_text = f.read()
            else:
                raise FileNotFoundError(
                    f"학습할 데이터 파일이 없습니다. {data_dir} 폴더에 txt를 넣어주세요."
                )

        # 1K 시퀀스 길이보다 전체 텍스트가 짧을 경우, 데이터셋 인덱스 오버플로우 크래시 방지를 위해 공백 패딩 가산
        if len(full_text) <= self.seq_length:
            needed = self.seq_length - len(full_text) + 5
            full_text = full_text + (" " * needed)

        self.encoded_tensor = self.tokenizer.encode(full_text)

    def __len__(self):
        # stride 단위로 슬라이딩
        return (len(self.encoded_tensor) - self.seq_length) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.encoded_tensor[start : start + self.seq_length]
        y = self.encoded_tensor[start + 1 : start + self.seq_length + 1]
        return x, y


if __name__ == "__main__":
    dataset    = KoJamoDataset(stride=10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    x, y = next(iter(dataloader))
    print(f"스트라이드 적용 후 데이터 수: {len(dataset)}")
    print(f"입력 x: {x.shape}, 타겟 y: {y.shape}")
