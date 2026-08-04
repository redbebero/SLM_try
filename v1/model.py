import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -------------------------------------------------------------------------
# 1. Base Components (DeepSeek Style)
# -------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
    
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)[:, :xq.shape[1], :, :]
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)[:, :xk.shape[1], :, :]
    
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, in_dim, bias=False)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, x):
        # Swish(w1(x)) * w3(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# -------------------------------------------------------------------------
# 2. DeepSeek MLA (Multi-head Latent Attention)
# -------------------------------------------------------------------------
class MLA(nn.Module):
    def __init__(self, dim, num_heads, latent_dim):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim # 압축 차원 (예: 64)

        # Query 압축 및 복원
        self.q_down = nn.Linear(dim, latent_dim, bias=False)
        self.q_up = nn.Linear(latent_dim, dim, bias=False)
        
        # KV 압축 및 복원 (MLA의 핵심: KV를 latent_dim 1개로 압축)
        self.kv_down = nn.Linear(dim, latent_dim, bias=False)
        self.k_up = nn.Linear(latent_dim, dim, bias=False)
        self.v_up = nn.Linear(latent_dim, dim, bias=False)
        
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, use_kv_cache=False, past_kv=None):
        B, S, D = x.shape
        
        # 1. 압축 (Latent Space)
        c_q = self.q_down(x)    # [B, S, latent_dim]
        c_kv = self.kv_down(x)  # [B, S, latent_dim]
        
        if use_kv_cache:
            if past_kv is not None:
                curr_c_kv = torch.cat([past_kv, c_kv], dim=1)
            else:
                curr_c_kv = c_kv
            new_kv = c_kv # Cache only the new token's c_kv
        else:
            curr_c_kv = c_kv
            new_kv = None
            
        # 2. 복원 (Up-projection)
        q = self.q_up(c_q).view(B, S, self.num_heads, self.head_dim)
        k = self.k_up(curr_c_kv).view(B, curr_c_kv.shape[1], self.num_heads, self.head_dim)
        v = self.v_up(curr_c_kv).view(B, curr_c_kv.shape[1], self.num_heads, self.head_dim)
        
        # Q, K 정규화 (Exploding Logits 방지 - DeepSeek-V4 구조 적용)
        q = F.layer_norm(q, (self.head_dim,))
        k = F.layer_norm(k, (self.head_dim,))
        
        # RoPE 적용
        if freqs_cis is not None:
            freqs_cos, freqs_sin = freqs_cis
            q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
            
        # Attention 연산
        q = q.transpose(1, 2) # [B, Heads, S, HeadDim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # PyTorch 2.0 SDPA
        out = F.scaled_dot_product_attention(q, k, v, is_causal=not use_kv_cache)
        
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        
        return self.out_proj(out), new_kv

# -------------------------------------------------------------------------
# 3. L-Module & H-Module (HRM 구조)
# -------------------------------------------------------------------------
class LModule(nn.Module):
    """빠른 계산과 토큰 간의 정보를 교환하는 L-Module (DeepSeek Transformer Block)"""
    def __init__(self, dim, num_heads, latent_dim, mlp_hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MLA(dim, num_heads, latent_dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_hidden_dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x, freqs_cis, use_kv_cache=False, past_kv=None):
        attn_out, new_kv = self.attn(self.norm1(x), freqs_cis, use_kv_cache, past_kv)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, new_kv

class HModule(nn.Module):
    """추상적 계획과 시간차원(순환)의 문맥을 유지하는 H-Module (Gated Recurrent Unit 스타일)"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.update_gate = nn.Linear(dim * 2, dim)
        self.reset_gate = nn.Linear(dim * 2, dim)
        self.memory_layer = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, h_prev, l_out):
        # h_prev: 이전 스텝의 잠재 상태 (Latent State)
        # l_out: 이번 스텝 L-Module의 연산 결과
        concat_input = torch.cat([h_prev, l_out], dim=-1)
        
        z = torch.sigmoid(self.update_gate(concat_input))
        r = torch.sigmoid(self.reset_gate(concat_input))
        
        h_candidate = torch.tanh(self.memory_layer(self.drop(torch.cat([r * h_prev, l_out], dim=-1))))
        h_new = (1 - z) * h_prev + z * h_candidate
        return h_new

# -------------------------------------------------------------------------
# 4. Micro HRM-DeepSeek Model
# -------------------------------------------------------------------------
class MicroHRMDeepSeek(nn.Module):
    def __init__(self, vocab_size=2000, dim=512, num_heads=8, latent_dim=64, 
                 mlp_hidden_dim=2048, max_seq_len=1024, thinking_steps=5):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.thinking_steps = thinking_steps
        self.max_seq_len = max_seq_len
        
        self.embed = nn.Embedding(vocab_size, dim)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim // num_heads, max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)
        
        # 단일 L-Module과 H-Module을 선언하고, 이를 T번 순환 재사용합니다 (Parameter Sharing).
        self.l_module = LModule(dim, num_heads, latent_dim, mlp_hidden_dim)
        self.h_module = HModule(dim)
        
        self.final_norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # 임베딩 가중치 공유 (Weight Tying)로 파라미터 극적 절약
        self.output.weight = self.embed.weight 

        print(f"Model initialized with {sum(p.numel() for p in self.parameters())/1e6:.2f}M parameters.")
        
    def forward(self, input_ids, state=None, use_kv_cache=False, past_kv=None):
        B, S = input_ids.shape
        
        if use_kv_cache and past_kv is not None:
            past_seq_len = past_kv.shape[1]
            freqs_cos = self.freqs_cos[past_seq_len : past_seq_len + S].to(input_ids.device)
            freqs_sin = self.freqs_sin[past_seq_len : past_seq_len + S].to(input_ids.device)
        else:
            freqs_cos = self.freqs_cos[:S].to(input_ids.device)
            freqs_sin = self.freqs_sin[:S].to(input_ids.device)
            
        freqs_cis = (freqs_cos, freqs_sin)
        
        x = self.embed(input_ids)
        if state is None:
            zH = x
            zL = x
        else:
            zH, zL = state
            
        # L runs thinking_steps times, H runs 1 time
        # 1-Step Gradient (Paper): All but last step use no_grad to save memory
        for i in range(self.thinking_steps):
            if i < self.thinking_steps - 1:
                with torch.no_grad():
                    combined_L = zL + zH + x
                    zL, _ = self.l_module(combined_L, freqs_cis, use_kv_cache, past_kv)
            else:
                combined_L = zL + zH + x
                zL, new_kv = self.l_module(combined_L, freqs_cis, use_kv_cache, past_kv)
            
        zH = self.h_module(zH, zL)
        state = (zH, zL)
            
        # 예측
        out = self.final_norm(zH)
        logits = self.output(out)
        
        if use_kv_cache:
            return state, logits, new_kv
        return state, logits

    def compute_log_probs(self, input_ids, labels, return_logits=False):
        """
        DPO 학습을 위해 특정 시퀀스의 Token-level Log Probability를 계산합니다.
        labels에서 -100인 부분(프롬프트)은 무시하고, 실제 정답/오답 토큰들의 확률만 합산합니다.
        """
        # 1. Forward Pass (KV Cache 없이 전체 시퀀스 처리)
        _, logits = self.forward(input_ids, state=None, use_kv_cache=False)
        
        # 2. Log Softmax로 확률 변환
        # logits: [B, S, V]
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 3. Label(-100 제외) 위치의 토큰 확률만 추출
        # labels: [B, S]
        # Mask out padding (-100)
        loss_mask = (labels != -100)
        
        # labels가 -100인 곳을 0으로 만들어 gather 시 인덱스 에러 방지
        safe_labels = labels.masked_fill(~loss_mask, 0)
        
        # 정답 토큰의 Log Prob 추출: [B, S]
        token_log_probs = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask 적용 (프롬프트 부분 및 패딩은 0으로 처리)
        token_log_probs = token_log_probs * loss_mask
        
        # 문장 전체의 Log Prob 합산: [B]
        seq_log_probs = token_log_probs.sum(dim=-1)
        
        if return_logits:
            return seq_log_probs, logits
        return seq_log_probs

# 테스트 스크립트
if __name__ == "__main__":
    # 6GB VRAM에 완벽히 맞는 설정: 약 5M~10M 파라미터 수준으로 더 줄이거나 늘릴 수 있습니다.
    model = MicroHRMDeepSeek(vocab_size=2000, dim=512, num_heads=8, latent_dim=64, mlp_hidden_dim=2048, thinking_steps=5)
    
    # 더미 입력 [Batch=2, Seq=128]
    dummy_input = torch.randint(0, 2000, (2, 128))
    
    # 순전파 테스트
    logits = model(dummy_input)
    print(f"Output shape: {logits.shape} (Expected: [2, 128, 2000])")
