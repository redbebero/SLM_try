import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
from tokenizer import KoJamoTokenizer


def sft_text_with_eos(text):
    """Append the tokenizer's newline EOS marker to every SFT answer."""
    return text if text.endswith("\n") else text + "\n"


class KoJamoDataset(Dataset):
    """
    텍스트 파일을 읽어서 (Seq_len, 6) 텐서로 변환하여
    Next-token 예측을 위한 (input, target) 텐서 쌍을 반환합니다.
    is_sft=True인 경우 지문 단위 개별 샘플로 로드하고 개별 토큰화합니다.
    """
    def __init__(self, data_dir="train_data", seq_length=50, stride=10, is_sft=False):
        self.tokenizer  = KoJamoTokenizer()
        self.seq_length = seq_length
        self.stride     = stride  # 속도 향상을 위한 건너뛰기 보폭
        self.is_sft     = is_sft

        if is_sft:
            cache_path = os.path.join(data_dir, "pretokenized_sft_cache.pt")
        else:
            cache_path = os.path.join(data_dir, "pretokenized_cache.pt")
            mmap_path = os.path.join(data_dir, "pretokenized_cache.bin")
            mmap_meta_path = os.path.join(data_dir, "pretokenized_cache.meta.json")

            if os.path.exists(mmap_path) and os.path.exists(mmap_meta_path):
                with open(mmap_meta_path, "r", encoding="utf-8") as meta_file:
                    meta = json.load(meta_file)
                # Old caches contain only token_count and allow windows to cross
                # source-sample boundaries. Rebuild them with sample offsets.
                if "sample_starts" in meta:
                    token_count = int(meta["token_count"])
                    self.sample_starts = np.asarray(meta["sample_starts"], dtype=np.int64)
                    self.encoded_tensor = np.memmap(
                        mmap_path, dtype=np.uint8, mode="r", shape=(token_count, 6)
                    )
                    print(
                        f"✅ 샘플 경계 캐시 로드 완료: {token_count:,} tokens / "
                        f"{len(self.sample_starts):,} windows"
                    )
                    return
                print("♻️ 기존 캐시는 샘플 경계 정보가 없어 재생성합니다.")

        if is_sft and os.path.exists(cache_path):
            print(f"🚀 캐시 파일 발견: {cache_path} 로드 중...")
            if is_sft:
                # SFT 캐시는 리스트 형태의 uint8 텐서 및 마스크 딕셔너리
                cache_data = torch.load(cache_path, map_location="cpu")
                if isinstance(cache_data, dict) and "samples" in cache_data:
                    self.samples = [s.long() for s in cache_data["samples"]]
                    self.loss_masks = [m.float() for m in cache_data["masks"]]
                else:
                    # 구 버전 캐시 예외 대처 (삭제하고 강제 재컴파일 유도)
                    print("🚨 구버전 캐시 포맷이 감지되어 기존 캐시를 자동 소거하고 재생성합니다.")
                    os.remove(cache_path)
                    raise FileNotFoundError("Rebuilding SFT cache with masks...")
                print(f"✅ SFT 캐시 로드 완료. 샘플 수: {len(self.samples)}")
        else:
            # train_data 폴더에서 모든 txt를 샘플 목록으로 수집합니다.
            import glob
            samples = []
            source_files = []
            
            if os.path.isdir(data_dir):
                txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
                # 캐시용 파일은 수집 대상에서 제외
                txt_files = [f for f in txt_files if not f.endswith("cache.pt")]
                
                if txt_files:
                    print(f"📖 데이터 폴더에서 텍스트 수집 완료: {len(txt_files)}개 파일 로드 중...")
                    source_files = sorted(txt_files)
                    if is_sft:
                        texts = []
                        for fp in source_files:
                            with open(fp, "r", encoding="utf-8") as f:
                                texts.append(f.read())
                    
                    if is_sft:
                        # SFT인 경우 개별 샘플(\n\n 기준)로 분할
                        raw_samples = [
                            sample.strip()
                            for text in texts
                            for sample in text.split("\n\n")
                            if sample.strip()
                        ]
                        
                        print(f"⚙️ SFT 개별 샘플 토큰화 및 프롬프트 마스킹 진행 중 (총 {len(raw_samples)}개)...")
                        self.samples = []
                        self.loss_masks = []
                        for s in raw_samples:
                            # Q: 와 A: 의 경계를 찾아 프롬프트 부분의 Loss 차단 마스크 생성
                            parts = s.split("\nA: ")
                            if len(parts) == 2:
                                prompt_text = parts[0] + "\nA: "
                            else:
                                prompt_text = "" # 분리 실패 시 전체 학습
                            
                            prompt_tokens = self.tokenizer.encode(prompt_text)
                            full_tokens = self.tokenizer.encode(sft_text_with_eos(s))
                            
                            L_prompt = len(prompt_tokens)
                            mask = torch.ones(len(full_tokens), dtype=torch.float32)
                            mask[:L_prompt] = 0.0 # 프롬프트 부분은 0.0으로 마스킹
                            
                            self.samples.append(full_tokens)
                            self.loss_masks.append(mask)
                        
                        print(f"💾 uint8 SFT 캐시 디스크 저장 중 -> {cache_path}")
                        saved_list = [s.to(torch.uint8) for s in self.samples]
                        saved_masks = [m.to(torch.uint8) for m in self.loss_masks]
                        torch.save({"samples": saved_list, "masks": saved_masks}, cache_path)
                        print("✅ SFT 캐시 디스크 저장 완료.")
                        
                        # long/float 형식으로 최종 보존
                        self.samples = [s.long() for s in self.samples]
                        self.loss_masks = [m.float() for m in self.loss_masks]
                        return
                    else:
                        pass
                
            # 만약 폴더에 텍스트가 없거나 경로가 폴더가 아닐 때의 예외 대체
            if not is_sft:
                if not source_files:
                    fallback = "datasets/wiki_clean_short.txt"
                    if os.path.exists(fallback):
                        print(f"⚠️ 폴더에 데이터가 없어 {fallback} 파일을 사용합니다.")
                        source_files = [fallback]
                    else:
                        raise FileNotFoundError(
                            f"학습할 데이터 파일이 없습니다. {data_dir} 폴더에 txt를 넣어주세요."
                        )

                print("⚙️ 샘플 단위 토큰화 진행 중 (디스크 매핑 모드)...")
                # Each source line is one sample. Allocate one contiguous memmap,
                # but record valid window starts so __getitem__ never crosses lines.
                def iter_samples():
                    for source in source_files:
                        with open(source, "r", encoding="utf-8") as f:
                            for line in f:
                                sample = line.strip()
                                if sample:
                                    yield sample

                total_chars = sum(len(sample) + 1 for sample in iter_samples())
                encoded_temp = np.memmap(
                    mmap_path, dtype=np.uint8, mode="w+", shape=(total_chars, 6)
                )
                sample_starts = []
                offset = 0
                sample_count = 0
                for index, sample in enumerate(iter_samples()):
                    encoded = np.asarray(self.tokenizer.encode(sample), dtype=np.uint8)
                    sample_length = len(encoded)
                    encoded_temp[offset : offset + sample_length] = encoded
                    # Need seq_length input tokens plus one target token.
                    if sample_length >= self.seq_length + 1:
                        last_start = offset + sample_length - self.seq_length - 1
                        sample_starts.extend(
                            range(offset, last_start + 1, self.stride)
                        )
                    offset += sample_length + 1  # reserve one delimiter position
                    sample_count = index + 1
                    if index == 0 or (index + 1) % 50_000 == 0:
                        print(
                            f"   {index + 1:,} samples",
                            flush=True,
                        )
                encoded_temp.flush()
                with open(mmap_meta_path, "w", encoding="utf-8") as meta_file:
                    json.dump(
                        {
                            "token_count": total_chars,
                            "sample_starts": sample_starts,
                        },
                        meta_file,
                    )
                print(f"✅ 메모리 매핑 캐시 저장 완료 -> {mmap_path}")
                self.encoded_tensor = encoded_temp
                self.sample_starts = np.asarray(sample_starts, dtype=np.int64)

    def __len__(self):
        if self.is_sft:
            return len(self.samples)
        return len(self.sample_starts)

    def __getitem__(self, idx):
        if self.is_sft:
            sample = self.samples[idx]
            mask = self.loss_masks[idx]
            # Next-token prediction 쌍 반환
            x = sample[:-1]
            y = sample[1:]
            mask_y = mask[1:]
            return x, y, mask_y
            
        start = int(self.sample_starts[idx])
        x = self.encoded_tensor[start : start + self.seq_length]
        y = self.encoded_tensor[start + 1 : start + self.seq_length + 1]
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.array(x, copy=True))
            y = torch.from_numpy(np.array(y, copy=True))
        x = x.long()
        y = y.long()
        return x, y


if __name__ == "__main__":
    dataset    = KoJamoDataset(stride=10, is_sft=True)
    print(f"SFT 모드 데이터 수: {len(dataset)}")
    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"샘플 0 입력 x: {x.shape}, 타겟 y: {y.shape}")
