import math
import torch
import torch.nn as nn


class KoJamoNet(nn.Module):
    """6-Track 자소 임베딩(초/중/종/기호/영어/숫자) + 표준 GRU 순환 코어.
    출력은 초성 -> 중성 -> 종성 순서로 앞선 예측(또는 정답)을 힌트 삼아
    다음 트랙을 예측하는 계단식(cascade) 디코딩 구조."""

    def __init__(self, vocab_sizes=(20, 22, 29, 36, 53, 11), emb_dim=64,
                 hidden_dim=256, num_layers=3, dropout=0.1, num_heads=8):
        super().__init__()
        n_cho, n_jung, n_jong, n_sym, n_eng, n_num = vocab_sizes

        # 1. 6-Track Embeddings
        self.emb_cho = nn.Embedding(n_cho, emb_dim, padding_idx=0)
        self.emb_jung = nn.Embedding(n_jung, emb_dim, padding_idx=0)
        self.emb_jong = nn.Embedding(n_jong, emb_dim, padding_idx=0)
        self.emb_sym = nn.Embedding(n_sym, emb_dim, padding_idx=0)
        self.emb_eng = nn.Embedding(n_eng, emb_dim, padding_idx=0)
        self.emb_num = nn.Embedding(n_num, emb_dim, padding_idx=0)

        # 2. Type Embedding
        self.emb_type = nn.Embedding(4, 32)

        # 3. Input Projection
        # +emb_dim: 직전 글자 종성(받침 유무/종류) 임베딩 — 조사(을/를, 은/는, 이/가) 형태가
        # 직전 받침 유무로 결정되는 한국어 형태음소 규칙을 GRU의 순환 기억에만 맡기지 않고
        # 명시적으로 직접 힌트를 줌 (초->중->종 계단식 힌트와 같은 철학을 시간축에 적용)
        self.proj_in = nn.Linear(emb_dim * 7 + 32, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 4. 표준 순환 코어 (인과적: GRU는 시퀀스를 왼쪽 -> 오른쪽으로만 처리하므로 미래 누출 없음)
        self.core = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        # 4-1. GRU 뒤에 얹는 causal self-attention 한 층 — GRU의 hidden state 압축(장거리 정보 병목)을
        # 보완하기 위해 모든 과거 위치에 직접 연결되는 통로를 추가 (사인파 위치인코딩은 학습파라미터 없이
        # 시퀀스 길이 무관하게 동작 — Stage1 seq64 / Stage2 seq1000 모두 대응)
        # pre-LN: attention 진입 전 정규화 — Transformer 학습 안정성의 표준 관행.
        # 이거 없이 LR 워밍업까지 빠뜨렸다가 attn.in_proj_weight가 학습 초반부터 폭주해 NaN 발산한 적 있음.
        self.pre_attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # 위치인코딩/causal mask 매 forward 재계산 낭비 제거 — 최대 길이(1000)까지 미리 계산해두고 슬라이싱만 함.
        # persistent=False: state_dict에 안 들어가서 체크포인트 호환성(구버전 로드)에 영향 없음.
        self._max_cached_len = 1000
        self.register_buffer(
            "_pos_cache", self._sinusoidal_position_encoding(self._max_cached_len, hidden_dim, "cpu"),
            persistent=False,
        )
        self.register_buffer(
            "_causal_mask_cache",
            torch.triu(torch.ones(self._max_cached_len, self._max_cached_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

        # 5. Type Classifier Head
        self.head_type = nn.Linear(hidden_dim, 4)

        # 6. Track Decoder Heads
        self.head_cho = nn.Linear(hidden_dim, n_cho)
        self.head_sym = nn.Linear(hidden_dim, n_sym)
        self.head_eng = nn.Linear(hidden_dim, n_eng)
        self.head_num = nn.Linear(hidden_dim, n_num)

        # 계단식 디코딩: 앞 트랙의 (정답 또는 예측) 임베딩을 힌트로 이어붙여 다음 트랙 예측
        self.head_jung = nn.Linear(hidden_dim + emb_dim, n_jung)
        self.head_jong = nn.Linear(hidden_dim + emb_dim * 2, n_jong)

    @staticmethod
    def _sinusoidal_position_encoding(seq_len, dim, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _get_types(self, x):
        sym = x[:, :, 3] > 0
        eng = x[:, :, 4] > 0
        num = x[:, :, 5] > 0

        types = torch.zeros(x.shape[0], x.shape[1], dtype=torch.long, device=x.device)
        types[sym] = 1
        types[eng] = 2
        types[num] = 3
        return types

    def forward(self, x, target_for_forcing=None):
        types = self._get_types(x)
        type_emb = self.emb_type(types)

        # 직전 타임스텝의 종성(받침) — 조사 형태 결정 규칙을 위한 명시적 힌트
        prev_jong = torch.cat([
            torch.zeros(x.size(0), 1, dtype=torch.long, device=x.device),
            x[:, :-1, 2]
        ], dim=1)
        prev_jong_emb = self.emb_jong(prev_jong)

        merged = torch.cat([
            self.emb_cho(x[:, :, 0]),
            self.emb_jung(x[:, :, 1]),
            self.emb_jong(x[:, :, 2]),
            self.emb_sym(x[:, :, 3]),
            self.emb_eng(x[:, :, 4]),
            self.emb_num(x[:, :, 5]),
            type_emb,
            prev_jong_emb
        ], dim=-1)

        core_input = self.dropout(self.proj_in(merged))
        core_state, _ = self.core(core_input)

        # causal self-attention: GRU가 압축한 hidden state로는 놓치기 쉬운 먼 거리 직접 참조 보완
        seq_len = core_state.size(1)
        if seq_len <= self._max_cached_len:
            pos = self._pos_cache[:seq_len]
            causal_mask = self._causal_mask_cache[:seq_len, :seq_len]
        else:
            # 캐시 범위 밖(1000자 초과)인 경우에만 예외적으로 즉석 계산
            pos = self._sinusoidal_position_encoding(seq_len, core_state.size(-1), core_state.device)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=core_state.device, dtype=torch.bool), diagonal=1
            )
        attn_input = self.pre_attn_norm(core_state + pos.unsqueeze(0))
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, attn_mask=causal_mask, need_weights=False)
        core_state = self.attn_norm(core_state + attn_out)

        logits_type = self.head_type(core_state)
        logits_cho = self.head_cho(core_state)
        logits_sym = self.head_sym(core_state)
        logits_eng = self.head_eng(core_state)
        logits_num = self.head_num(core_state)

        if target_for_forcing is not None:
            forced_cho_emb = self.emb_cho(target_for_forcing[:, :, 0])
            forced_jung_emb = self.emb_jung(target_for_forcing[:, :, 1])
            logits_jung = self.head_jung(torch.cat([core_state, forced_cho_emb], dim=-1))
            logits_jong = self.head_jong(torch.cat([core_state, forced_cho_emb, forced_jung_emb], dim=-1))
        else:
            pred_cho_emb = self.emb_cho(logits_cho.argmax(dim=-1))
            logits_jung = self.head_jung(torch.cat([core_state, pred_cho_emb], dim=-1))
            pred_jung_emb = self.emb_jung(logits_jung.argmax(dim=-1))
            logits_jong = self.head_jong(torch.cat([core_state, pred_cho_emb, pred_jung_emb], dim=-1))

        return logits_type, logits_cho, logits_jung, logits_jong, logits_sym, logits_eng, logits_num
