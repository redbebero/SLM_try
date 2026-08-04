"""Small Korean-jamo HRM-lite: fast L state + slow H state."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HRMJamoNet(nn.Module):
    """A compact fixed-compute hierarchical recurrent jamo model.

    L updates at every character position. H updates every ``cycle_steps``
    positions. Repeated segment calls refine the same sequence with detached
    states, matching HRM's one-step/deep-supervision training style.
    """

    def __init__(self, vocab_sizes=(20, 22, 29, 36, 53, 11), emb_dim=32,
                 hidden_dim=256, cycle_steps=4, use_attention=False,
                 use_prompt_memory=False):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.cycle_steps = cycle_steps
        self.use_attention = use_attention
        self.use_prompt_memory = use_prompt_memory
        n_cho, n_jung, n_jong, n_sym, n_eng, n_num = vocab_sizes

        self.emb_cho = nn.Embedding(n_cho, emb_dim, padding_idx=0)
        self.emb_jung = nn.Embedding(n_jung, emb_dim, padding_idx=0)
        self.emb_jong = nn.Embedding(n_jong, emb_dim, padding_idx=0)
        self.emb_sym = nn.Embedding(n_sym, emb_dim, padding_idx=0)
        self.emb_eng = nn.Embedding(n_eng, emb_dim, padding_idx=0)
        self.emb_num = nn.Embedding(n_num, emb_dim, padding_idx=0)
        self.emb_type = nn.Embedding(4, emb_dim)
        self.input_proj = nn.Linear(emb_dim * 7, hidden_dim)

        self.l_cell = nn.GRUCell(hidden_dim + hidden_dim, hidden_dim)
        self.h_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.l_norm = nn.LayerNorm(hidden_dim)
        self.h_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim * 2)
        if use_attention:
            self.context_attn = nn.MultiheadAttention(
                hidden_dim * 2, num_heads=4, batch_first=True,
            )
            self.context_norm = nn.LayerNorm(hidden_dim * 2)
        if use_prompt_memory:
            self.prompt_proj = nn.Linear(hidden_dim, hidden_dim * 2)
            self.prompt_attn = nn.MultiheadAttention(
                hidden_dim * 2, num_heads=4, batch_first=True,
            )
            self.prompt_norm = nn.LayerNorm(hidden_dim * 2)

        self.head_type = nn.Linear(hidden_dim * 2, 4)
        self.head_cho = nn.Linear(hidden_dim * 2, n_cho)
        self.head_jung = nn.Linear(hidden_dim * 2, n_jung)
        self.head_jong = nn.Linear(hidden_dim * 2, n_jong)
        self.head_sym = nn.Linear(hidden_dim * 2, n_sym)
        self.head_eng = nn.Linear(hidden_dim * 2, n_eng)
        self.head_num = nn.Linear(hidden_dim * 2, n_num)

    @staticmethod
    def _get_types(x):
        return ((x[:, :, 3] > 0).long()
                + 2 * (x[:, :, 4] > 0).long()
                + 3 * (x[:, :, 5] > 0).long())

    def encode_input(self, x):
        types = self._get_types(x)
        prev_jong = torch.cat([
            torch.zeros(x.size(0), 1, dtype=torch.long, device=x.device),
            x[:, :-1, 2],
        ], dim=1)
        return self.input_proj(torch.cat([
            self.emb_cho(x[:, :, 0]), self.emb_jung(x[:, :, 1]),
            self.emb_jong(prev_jong), self.emb_sym(x[:, :, 3]),
            self.emb_eng(x[:, :, 4]), self.emb_num(x[:, :, 5]),
            self.emb_type(types),
        ], dim=-1))

    def init_state(self, batch_size, device):
        zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
        return zeros.clone(), zeros.clone()

    def forward_segment(self, x, state=None, prompt_mask=None):
        inputs = self.encode_input(x)
        batch, seq_len, _ = inputs.shape
        if state is None:
            h, l = self.init_state(batch, x.device)
        else:
            h, l = state
        outputs = []
        for index in range(seq_len):
            l = self.l_norm(self.l_cell(torch.cat([inputs[:, index], h], dim=-1), l))
            if (index + 1) % self.cycle_steps == 0:
                h = self.h_norm(self.h_cell(l, h))
            outputs.append(torch.cat([h, l], dim=-1))
        hidden = self.output_norm(torch.stack(outputs, dim=1))
        if self.use_attention:
            length = hidden.size(1)
            causal_mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1
            )
            context, _ = self.context_attn(hidden, hidden, hidden, attn_mask=causal_mask)
            hidden = self.context_norm(hidden + context)
        if self.use_prompt_memory:
            if prompt_mask is None:
                prompt_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=x.device)
            memory = self.prompt_proj(inputs)
            key_padding_mask = ~prompt_mask
            context, _ = self.prompt_attn(
                hidden, memory, memory, key_padding_mask=key_padding_mask,
            )
            hidden = self.prompt_norm(hidden + context)
        logits = (
            self.head_type(hidden), self.head_cho(hidden), self.head_jung(hidden),
            self.head_jong(hidden), self.head_sym(hidden), self.head_eng(hidden),
            self.head_num(hidden),
        )
        return logits, (h, l)

    def forward(self, x, state=None, segments=1, prompt_mask=None):
        logits, state = self.forward_segment(x, state, prompt_mask=prompt_mask)
        for _ in range(segments - 1):
            state = tuple(item.detach() for item in state)
            logits, state = self.forward_segment(x, state, prompt_mask=prompt_mask)
        return logits, state


class HRMConditionalNet(HRMJamoNet):
    """Prompt-conditioned HRM decoder.

    The prompt is consumed first to build the H/L state.  The answer then
    continues from that state autoregressively.  Unlike the old sequence
    head, this keeps the question representation separate from answer-side
    teacher-forcing features while retaining the six jamo tracks.
    """

    def forward_segment(self, x, state=None, prompt_mask=None):
        batch, seq_len, _ = x.shape
        inputs = self.encode_input(x)
        if prompt_mask is None:
            prompt_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=x.device)
        starts = prompt_mask.long().sum(dim=1).clamp(max=seq_len)
        if state is None:
            h, l = self.init_state(batch, x.device)
        else:
            h, l = state
        steps = torch.zeros(batch, dtype=torch.long, device=x.device)
        outputs = []
        for index in range(seq_len):
            l_new = self.l_norm(self.l_cell(torch.cat([inputs[:, index], h], dim=-1), l))
            l = l_new
            steps = steps + 1
            update_h = (steps % self.cycle_steps == 0)
            h_new = self.h_norm(self.h_cell(l, h))
            h = torch.where(update_h.unsqueeze(-1), h_new, h)
            outputs.append(torch.cat([h, l], dim=-1))
        hidden = self.output_norm(torch.stack(outputs, dim=1))
        raw_logits = (
            self.head_type(hidden), self.head_cho(hidden), self.head_jung(hidden),
            self.head_jong(hidden), self.head_sym(hidden), self.head_eng(hidden),
            self.head_num(hidden),
        )
        logits = tuple(F.log_softmax(item, dim=-1) for item in raw_logits)
        # Keep the boundary available for diagnostics and future decoding.
        self.last_prompt_starts = starts.detach()
        return logits, (h, l)


class HRMContextNet(HRMJamoNet):
    """Causal context encoder followed by hierarchical H/L refinement.

    The encoder preserves long prompt information; the recurrent H/L states
    still perform fixed-compute refinement over the encoder features.  Inputs
    remain six-track jamo tokens, so this is a hybrid HRM rather than a plain
    character Transformer.
    """

    def __init__(self, vocab_sizes=(20, 22, 29, 36, 53, 11), emb_dim=32,
                 hidden_dim=256, cycle_steps=4, context_layers=2,
                 context_heads=4, context_dropout=0.0,
                 max_seq_length=512, use_copy=False,
                 use_current_jong=False, use_joint_jamo=False,
                 joint_rank=32, use_query_summary=False,
                 use_char_head=False, char_vocab_size=11172):
        super().__init__(vocab_sizes=vocab_sizes, emb_dim=emb_dim,
                         hidden_dim=hidden_dim, cycle_steps=cycle_steps)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=context_heads,
            dim_feedforward=hidden_dim * 4, dropout=context_dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(layer, num_layers=context_layers)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_length, hidden_dim)
        self.context_skip = nn.Linear(hidden_dim, hidden_dim * 2)
        with torch.no_grad():
            self.context_skip.weight.zero_()
            self.context_skip.weight[:hidden_dim].copy_(torch.eye(hidden_dim))
            self.context_skip.bias.zero_()
        self.use_copy = use_copy
        self.use_current_jong = use_current_jong
        self.use_query_summary = use_query_summary
        self.use_char_head = use_char_head
        self.char_vocab_size = char_vocab_size
        if use_query_summary:
            # Causal prefix summary. At the first answer position this is a
            # compact question state; later positions also include generated
            # answer context. No future token can enter the summary.
            self.query_summary_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.use_answer_start_head = use_query_summary
        if self.use_answer_start_head:
            self.answer_start_heads = nn.ModuleList([
                nn.Linear(hidden_dim * 2, size) for size in (4, *vocab_sizes)
            ])
        self.last_answer_start_logits = None
        if use_current_jong:
            self.current_jong_proj = nn.Linear(emb_dim, hidden_dim)
        self.use_joint_jamo = use_joint_jamo
        self.last_joint_logits = None
        # Standalone compatibility jamo (e.g. ``ㄴ``) is tokenized on the
        # choseong track, while an answer's final consonant belongs to the
        # jongseong track.  This fixed Korean relation lets the copy path
        # transfer F=ㄴ/ㄹ/... without adding a large character vocabulary.
        self.register_buffer("cho_to_jong", torch.tensor(
            [0, 1, 2, 4, 7, 0, 8, 16, 17, 0, 19, 20, 21, 22, 0, 23, 24, 25, 26, 27],
            dtype=torch.long,
        ), persistent=False)
        if use_joint_jamo:
            self.joint_query = nn.Linear(hidden_dim * 2, joint_rank)
            self.joint_cho = nn.Embedding(vocab_sizes[0], joint_rank, padding_idx=0)
            self.joint_jung = nn.Embedding(vocab_sizes[1], joint_rank, padding_idx=0)
            self.joint_jong = nn.Embedding(vocab_sizes[2], joint_rank, padding_idx=0)
        if use_copy:
            self.copy_query = nn.Linear(hidden_dim * 2, hidden_dim)
            self.copy_gate = nn.Linear(hidden_dim * 2, len(vocab_sizes) + 1)
        output_dim = hidden_dim * 2
        self.base_cho = nn.Linear(output_dim, vocab_sizes[0])
        self.base_jung = nn.Linear(output_dim, vocab_sizes[1])
        self.base_jong = nn.Linear(output_dim, vocab_sizes[2])
        self.jamo_fusion = nn.Linear(output_dim + emb_dim * 3, output_dim)
        self.jamo_norm = nn.LayerNorm(output_dim)
        if use_char_head:
            self.char_head = nn.Linear(output_dim, char_vocab_size)
        self.last_char_logits = None
        self.max_seq_length = max_seq_length
        self.context_layers = context_layers
        self.context_heads = context_heads

    def forward_segment(self, x, state=None, prompt_mask=None):
        del prompt_mask
        inputs = self.encode_input(x)
        if self.use_current_jong:
            inputs = inputs + self.current_jong_proj(self.emb_jong(x[:, :, 2]))
        seq_len = inputs.size(1)
        if seq_len > self.max_seq_length:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_length")
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        inputs = inputs + self.pos_emb(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        inputs = self.context_norm(self.context_encoder(inputs, mask=causal_mask))
        batch = inputs.size(0)
        if state is None:
            h, l = self.init_state(batch, x.device)
        else:
            h, l = state
        outputs = []
        for index in range(seq_len):
            l = self.l_norm(self.l_cell(torch.cat([inputs[:, index], h], dim=-1), l))
            if (index + 1) % self.cycle_steps == 0:
                h = self.h_norm(self.h_cell(l, h))
            outputs.append(torch.cat([h, l], dim=-1))
        hidden = self.output_norm(torch.stack(outputs, dim=1))
        hidden = hidden + self.context_skip(inputs)
        if self.use_query_summary:
            prefix_count = torch.arange(
                1, seq_len + 1, device=x.device, dtype=inputs.dtype
            ).view(1, -1, 1)
            prefix_summary = inputs.cumsum(dim=1) / prefix_count
            hidden = hidden + self.query_summary_proj(prefix_summary)
            if self.use_answer_start_head:
                start_hidden = self.jamo_norm(hidden)
                self.last_answer_start_logits = tuple(
                    F.log_softmax(head(start_hidden), dim=-1)
                    for head in self.answer_start_heads
                )
        p_cho = torch.softmax(self.base_cho(hidden), dim=-1)
        p_jung = torch.softmax(self.base_jung(hidden), dim=-1)
        p_jong = torch.softmax(self.base_jong(hidden), dim=-1)
        jamo_context = torch.cat([
            torch.matmul(p_cho, self.emb_cho.weight),
            torch.matmul(p_jung, self.emb_jung.weight),
            torch.matmul(p_jong, self.emb_jong.weight),
        ], dim=-1)
        hidden = self.jamo_norm(
            hidden + self.jamo_fusion(torch.cat([hidden, jamo_context], dim=-1))
        )
        self.last_char_logits = (
            F.log_softmax(self.char_head(hidden), dim=-1)
            if self.use_char_head else None
        )
        raw_logits = [
            self.head_type(hidden), self.head_cho(hidden), self.head_jung(hidden),
            self.head_jong(hidden), self.head_sym(hidden), self.head_eng(hidden),
            self.head_num(hidden),
        ]
        if self.use_joint_jamo:
            query = F.normalize(self.joint_query(hidden), dim=-1, eps=1e-6)
            cho_factors = F.normalize(self.joint_cho.weight, dim=-1, eps=1e-6)
            jung_factors = F.normalize(self.joint_jung.weight, dim=-1, eps=1e-6)
            jong_factors = F.normalize(self.joint_jong.weight, dim=-1, eps=1e-6)
            joint_scores = torch.einsum(
                "blr,cr,jr,gr->blcjg", query,
                cho_factors, jung_factors, jong_factors,
            )
            invalid = torch.zeros_like(joint_scores, dtype=torch.bool)
            invalid[:, :, 0, :, :] = True
            invalid[:, :, :, 0, :] = True
            # Use a large finite penalty instead of -inf: marginalizing an
            # entirely invalid PAD slice through logsumexp would create NaN
            # gradients even though the forward loss is finite.
            joint_scores = joint_scores.masked_fill(invalid, -1e4)
            self.last_joint_logits = joint_scores.detach()
            joint_log_probs = F.log_softmax(
                joint_scores.reshape(joint_scores.size(0), joint_scores.size(1), -1), dim=-1
            ).reshape_as(joint_scores)
            raw_logits[1] = torch.logsumexp(joint_log_probs, dim=(3, 4))
            raw_logits[2] = torch.logsumexp(joint_log_probs, dim=(2, 4))
            raw_logits[3] = torch.logsumexp(joint_log_probs, dim=(2, 3))
        if self.use_copy:
            query = self.copy_query(hidden)
            scores = torch.bmm(query, inputs.transpose(1, 2)) / (self.hidden_dim ** 0.5)
            future = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
            )
            scores = scores.masked_fill(future.unsqueeze(0), float("-inf"))
            copy_weights = torch.softmax(scores, dim=-1)
            types = self._get_types(x)
            standalone = (x[:, :, 0] > 0) & (x[:, :, 1:].sum(dim=-1) == 0)
            mapped_jong = self.cho_to_jong[x[:, :, 0]]
            jong_copy = torch.where(
                standalone & (mapped_jong > 0), mapped_jong, x[:, :, 2]
            )
            copy_values = [types, x[:, :, 0], x[:, :, 1], x[:, :, 2],
                           x[:, :, 3], x[:, :, 4], x[:, :, 5]]
            copy_values[3] = jong_copy
            gates = torch.sigmoid(self.copy_gate(hidden)).unbind(-1)
            mixed = []
            for prediction, values, gate in zip(raw_logits, copy_values, gates):
                copy_dist = torch.bmm(
                    copy_weights,
                    F.one_hot(values, num_classes=prediction.size(-1)).float(),
                )
                model_dist = torch.softmax(prediction, dim=-1)
                mixed.append(torch.log(
                    (1.0 - gate.unsqueeze(-1)) * model_dist
                    + gate.unsqueeze(-1) * copy_dist + 1e-8
                ))
            logits = tuple(mixed)
        else:
            logits = tuple(F.log_softmax(item, dim=-1) for item in raw_logits)
        return logits, (h, l)
