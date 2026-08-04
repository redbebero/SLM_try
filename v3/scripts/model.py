import torch
import torch.nn as nn
import torch.nn.functional as F

class KoJamoNet(nn.Module):
    def __init__(self, vocab_sizes=(20, 22, 29, 36, 53, 11), emb_dim=64,
                 hidden_dim=256, num_layers=4, dropout=0.1, cascade=True):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.cascade = cascade
        n_cho, n_jung, n_jong, n_sym, n_eng, n_num = vocab_sizes

        self.emb_cho = nn.Embedding(n_cho, emb_dim, padding_idx=0)
        self.emb_jung = nn.Embedding(n_jung, emb_dim, padding_idx=0)
        self.emb_jong = nn.Embedding(n_jong, emb_dim, padding_idx=0)
        self.emb_sym = nn.Embedding(n_sym, emb_dim, padding_idx=0)
        self.emb_eng = nn.Embedding(n_eng, emb_dim, padding_idx=0)
        self.emb_num = nn.Embedding(n_num, emb_dim, padding_idx=0)
        self.emb_type = nn.Embedding(4, 32)

        # Six track embeddings plus type embedding. Previous jong is included
        # once; duplicating it biased the core toward jong features.
        self.proj_in = nn.Linear(emb_dim * 6 + 32, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.core = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.head_type = nn.Linear(hidden_dim, 4)
        
        self.head_cho = nn.Linear(hidden_dim, n_cho)
        if cascade:
            self.head_jung = nn.Linear(hidden_dim + emb_dim, n_jung)
            self.head_jong = nn.Linear(hidden_dim + emb_dim * 2, n_jong)
        else:
            self.head_jung = nn.Linear(hidden_dim, n_jung)
            self.head_jong = nn.Linear(hidden_dim, n_jong)
        
        self.head_sym = nn.Linear(hidden_dim, n_sym)
        self.head_eng = nn.Linear(hidden_dim, n_eng)
        self.head_num = nn.Linear(hidden_dim, n_num)

    def _get_types(self, x):
        sym = (x[:, :, 3] > 0).long()
        eng = (x[:, :, 4] > 0).long()
        num = (x[:, :, 5] > 0).long()
        types = sym * 1 + eng * 2 + num * 3
        return types

    def forward(self, x, target_for_forcing=None, teacher_forcing_ratio=None):
        types = self._get_types(x)
        type_emb = self.emb_type(types)

        prev_jong = torch.cat([
            torch.zeros(x.size(0), 1, dtype=torch.long, device=x.device),
            x[:, :-1, 2]
        ], dim=1)
        prev_jong_emb = self.emb_jong(prev_jong)

        merged = torch.cat([
            self.emb_cho(x[:, :, 0]), self.emb_jung(x[:, :, 1]), prev_jong_emb,
            self.emb_sym(x[:, :, 3]), self.emb_eng(x[:, :, 4]), self.emb_num(x[:, :, 5]),
            type_emb
        ], dim=-1)

        core_input = self.dropout(self.proj_in(merged))
        
        core_state, _ = self.core(core_input)

        logits_type = self.head_type(core_state)
        logits_sym = self.head_sym(core_state)
        logits_eng = self.head_eng(core_state)
        logits_num = self.head_num(core_state)
        
        logits_cho = self.head_cho(core_state)

        if not self.cascade:
            log_type = F.log_softmax(logits_type, dim=-1)
            log_cho = F.log_softmax(logits_cho, dim=-1)
            log_jung = F.log_softmax(self.head_jung(core_state), dim=-1)
            log_jong = F.log_softmax(self.head_jong(core_state), dim=-1)
            return (log_type, log_cho, log_jung, log_jong,
                    F.log_softmax(logits_sym, dim=-1),
                    F.log_softmax(logits_eng, dim=-1),
                    F.log_softmax(logits_num, dim=-1))

        # Cascade Decoding for Hangul (Cho -> Jung -> Jong)
        
        p_cho = torch.softmax(logits_cho, dim=-1)
        predicted_cho_emb = torch.matmul(p_cho, self.emb_cho.weight)
        if target_for_forcing is not None and teacher_forcing_ratio != 0.0:
            forced_cho_emb = self.emb_cho(target_for_forcing[:, :, 0])
            ratio = 1.0 if teacher_forcing_ratio is None else teacher_forcing_ratio
            force_mask = torch.rand(x.shape[:2], device=x.device) < ratio
            cho_emb = torch.where(force_mask.unsqueeze(-1), forced_cho_emb, predicted_cho_emb)
        else:
            cho_emb = predicted_cho_emb
            
        logits_jung = self.head_jung(torch.cat([core_state, cho_emb], dim=-1))
        
        p_jung = torch.softmax(logits_jung, dim=-1)
        predicted_jung_emb = torch.matmul(p_jung, self.emb_jung.weight)
        if target_for_forcing is not None and teacher_forcing_ratio != 0.0:
            forced_jung_emb = self.emb_jung(target_for_forcing[:, :, 1])
            ratio = 1.0 if teacher_forcing_ratio is None else teacher_forcing_ratio
            force_mask = torch.rand(x.shape[:2], device=x.device) < ratio
            jung_emb = torch.where(force_mask.unsqueeze(-1), forced_jung_emb, predicted_jung_emb)
        else:
            jung_emb = predicted_jung_emb
            
        logits_jong = self.head_jong(torch.cat([core_state, cho_emb, jung_emb], dim=-1))

        # Return log_softmax directly to be compatible with NLLLoss in train.py
        log_type = F.log_softmax(logits_type, dim=-1)
        log_cho = F.log_softmax(logits_cho, dim=-1)
        log_jung = F.log_softmax(logits_jung, dim=-1)
        log_jong = F.log_softmax(logits_jong, dim=-1)
        log_sym = F.log_softmax(logits_sym, dim=-1)
        log_eng = F.log_softmax(logits_eng, dim=-1)
        log_num = F.log_softmax(logits_num, dim=-1)

        return log_type, log_cho, log_jung, log_jong, log_sym, log_eng, log_num


class KoJamoTransformer(nn.Module):
    """Causal language model with differentiable mutual jamo fusion."""

    def __init__(self, vocab_sizes=(20, 22, 29, 36, 53, 11), emb_dim=64,
                 hidden_dim=256, num_layers=4, num_heads=4, dropout=0.1,
                 max_seq_length=512):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        n_cho, n_jung, n_jong, n_sym, n_eng, n_num = vocab_sizes

        self.emb_cho = nn.Embedding(n_cho, emb_dim, padding_idx=0)
        self.emb_jung = nn.Embedding(n_jung, emb_dim, padding_idx=0)
        self.emb_jong = nn.Embedding(n_jong, emb_dim, padding_idx=0)
        self.emb_sym = nn.Embedding(n_sym, emb_dim, padding_idx=0)
        self.emb_eng = nn.Embedding(n_eng, emb_dim, padding_idx=0)
        self.emb_num = nn.Embedding(n_num, emb_dim, padding_idx=0)
        self.emb_type = nn.Embedding(4, 32)
        self.pos_emb = nn.Embedding(max_seq_length, hidden_dim)
        self.proj_in = nn.Linear(emb_dim * 6 + 32, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.core = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # First-pass probabilities provide soft jamo embeddings. The fusion
        # layer lets cho/jung/jong influence each other without argmax error
        # propagation from a hard cascade.
        self.base_cho = nn.Linear(hidden_dim, n_cho)
        self.base_jung = nn.Linear(hidden_dim, n_jung)
        self.base_jong = nn.Linear(hidden_dim, n_jong)
        self.fusion_proj = nn.Linear(hidden_dim + emb_dim * 3, hidden_dim)
        self.head_type = nn.Linear(hidden_dim, 4)
        self.head_cho = nn.Linear(hidden_dim, n_cho)
        self.head_jung = nn.Linear(hidden_dim, n_jung)
        self.head_jong = nn.Linear(hidden_dim, n_jong)
        self.head_sym = nn.Linear(hidden_dim, n_sym)
        self.head_eng = nn.Linear(hidden_dim, n_eng)
        self.head_num = nn.Linear(hidden_dim, n_num)

    def _get_types(self, x):
        sym = (x[:, :, 3] > 0).long()
        eng = (x[:, :, 4] > 0).long()
        num = (x[:, :, 5] > 0).long()
        return sym * 1 + eng * 2 + num * 3

    def forward(self, x, target_for_forcing=None, teacher_forcing_ratio=None):
        del target_for_forcing, teacher_forcing_ratio
        types = self._get_types(x)
        prev_jong = torch.cat([
            torch.zeros(x.size(0), 1, dtype=torch.long, device=x.device),
            x[:, :-1, 2],
        ], dim=1)
        merged = torch.cat([
            self.emb_cho(x[:, :, 0]), self.emb_jung(x[:, :, 1]),
            self.emb_jong(prev_jong), self.emb_sym(x[:, :, 3]),
            self.emb_eng(x[:, :, 4]), self.emb_num(x[:, :, 5]),
            self.emb_type(types),
        ], dim=-1)
        seq_len = x.size(1)
        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_length")
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.dropout(self.proj_in(merged) + self.pos_emb(positions))
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        h = self.norm(self.core(h, mask=causal_mask))

        p_cho = torch.softmax(self.base_cho(h), dim=-1)
        p_jung = torch.softmax(self.base_jung(h), dim=-1)
        p_jong = torch.softmax(self.base_jong(h), dim=-1)
        jamo_context = torch.cat([
            torch.matmul(p_cho, self.emb_cho.weight),
            torch.matmul(p_jung, self.emb_jung.weight),
            torch.matmul(p_jong, self.emb_jong.weight),
        ], dim=-1)
        fused = self.norm(h + self.fusion_proj(torch.cat([h, jamo_context], dim=-1)))
        logits = (
            self.head_type(fused), self.head_cho(fused), self.head_jung(fused),
            self.head_jong(fused), self.head_sym(fused), self.head_eng(fused),
            self.head_num(fused),
        )
        return tuple(F.log_softmax(item, dim=-1) for item in logits)
