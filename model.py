import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from dataset import PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX, special_symbols

BLANK_IDX = BOS_IDX


def exclude_from_muon(module: nn.Module):
    for param in module.parameters():
        param.is_muon_excluded = True


class Joint(nn.Module):
    def __init__(
        self,
        num_phonemes,
        d_enc,
        d_dec,
        d_hidden,
        **kwargs,
    ):
        super().__init__()
        self.forward_layer = nn.Linear(d_enc + d_dec, d_hidden, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(d_hidden, num_phonemes, bias=True)
        torch.nn.init.xavier_uniform_(self.forward_layer.weight)
        torch.nn.init.xavier_uniform_(self.project_layer.weight)
        exclude_from_muon(self.project_layer)

    def forward(self, enc_state: Tensor, dec_state: Tensor, t_dec: Tensor = None):
        assert enc_state.dim() == 3
        assert dec_state.dim() == 3
        d_enc = enc_state.size(-1)
        w_enc = self.forward_layer.weight[:, :d_enc]
        w_dec = self.forward_layer.weight[:, d_enc:]
        enc_proj = F.linear(enc_state, w_enc, self.forward_layer.bias).unsqueeze(2)
        dec_proj = F.linear(dec_state, w_dec).unsqueeze(1)
        outputs = self.tanh(enc_proj + dec_proj)
        return self.project_layer(outputs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor, offset: Tensor = None) -> Tensor:
        if offset is not None:
            assert x.size(1) == 1
            x = x + torch.gather(
                self.pe.expand(x.size(0), -1, -1), 1, offset.unsqueeze(2)
            )
        else:
            x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MHAJoint(nn.Module):
    def __init__(
        self,
        num_phonemes,
        d_enc,
        d_dec,
        d_hidden,
        num_heads=4,
        dropout=0.1,
        max_len=48,
        **kwargs,
    ):
        super().__init__()
        self.enc_pe = PositionalEncoding(d_enc, dropout, max_len)
        self.dec_pe = PositionalEncoding(d_dec, dropout, max_len)
        self.q_proj = nn.Linear(d_dec, d_dec)
        self.k_proj = nn.Linear(d_enc, d_dec)
        self.v_proj = nn.Linear(d_enc, d_dec)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_dec,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(d_dec)
        self.forward_layer = nn.Linear(d_enc + d_dec, d_hidden)
        self.project_layer = nn.Linear(d_hidden, num_phonemes, bias=True)

        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        torch.nn.init.xavier_uniform_(self.v_proj.weight)
        torch.nn.init.xavier_uniform_(self.project_layer.weight)
        exclude_from_muon(self.project_layer)

    def forward(self, enc_state: Tensor, dec_state: Tensor, t_dec: Tensor = None):
        dec_with_pe = self.dec_pe(dec_state, t_dec)
        q = self.q_proj(dec_with_pe)
        enc_with_pe = self.enc_pe(enc_state)
        k = self.k_proj(enc_with_pe)
        v = self.v_proj(enc_with_pe)
        context, _ = self.attn(q, k, v)
        context = self.ln(context + dec_with_pe)

        d_enc = enc_state.size(-1)
        w_enc = self.forward_layer.weight[:, :d_enc]
        w_dec = self.forward_layer.weight[:, d_enc:]
        enc_proj = F.linear(enc_state, w_enc, self.forward_layer.bias).unsqueeze(2)
        ctx_proj = F.linear(context, w_dec).unsqueeze(1)

        outputs = torch.tanh(enc_proj + ctx_proj)
        return self.project_layer(outputs)


class LstmEncoder(nn.Module):
    def __init__(
        self,
        num_graphemes: int,
        d_model: int,
        d_hidden: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.emb = nn.Embedding(num_graphemes, d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_actv = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_hidden // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(d_hidden, d_model)

        torch.nn.init.xavier_uniform_(self.conv1d.weight)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        exclude_from_muon(self.emb)

    def forward(self, input: Tensor):
        x = self.emb(input)
        x_conv = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        x_conv = self.conv_norm(x_conv)
        x_conv = self.conv_actv(x_conv)
        x = x + x_conv
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(
        self,
        num_phonemes: int,
        d_model: int,
        d_hidden: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.emb = nn.Embedding(num_phonemes, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(d_hidden, d_model)

        torch.nn.init.xavier_uniform_(self.fc.weight)
        exclude_from_muon(self.emb)

    def forward(self, input: Tensor):
        x = self.emb(input)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    def step(self, input: Tensor, h: Tensor, c: Tensor):
        x = self.emb(input)
        x, (h, c) = self.lstm(x, (h, c))
        x = self.fc(x)
        return x, h, c


class G2p(nn.Module):
    def __init__(
        self,
        max_len: int,
        encoder: nn.Module,
        decoder: nn.Module,
        joint: nn.Module,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        enc_state = self.encoder(src)
        dec_state = self.decoder(tgt)
        return self.joint(enc_state, dec_state)

    def encode(self, src: Tensor):
        device = src.device
        B = src.shape[0]
        enc_state = self.encoder(src)
        tgt = torch.full((B, self.max_len), BOS_IDX, dtype=torch.int32, device=device)
        hc = torch.zeros(
            (2, self.decoder.num_layers, B, self.decoder.d_hidden),
            dtype=torch.float32,
            device=device,
        )
        ts = torch.zeros((B, 2), dtype=torch.int32, device=device)
        return tgt, enc_state, hc, ts

    def decode_step(self, tgt, enc_state, src_lengths, hc, ts):
        h, c = hc[0], hc[1]
        t_src, t_tgt = ts[:, 0], ts[:, 1]
        B = tgt.size(0)
        input = tgt.gather(dim=1, index=t_tgt.unsqueeze(1))
        step_dec_state, new_h, new_c = self.decoder.step(input, h, c)
        end_of_src = torch.ge(t_src, src_lengths)  # (B,)
        safe_t_src = torch.min(t_src, src_lengths - 1)
        gathered_enc = enc_state.gather(
            dim=1, index=safe_t_src.view(B, 1, 1).expand(B, 1, enc_state.size(-1))
        )
        step_enc_state = torch.where(
            end_of_src.view(B, 1, 1),
            torch.zeros_like(gathered_enc),
            gathered_enc,
        )

        logits = self.joint(step_enc_state, step_dec_state, t_tgt.unsqueeze(1))
        pred = torch.argmax(logits, dim=-1).squeeze(1).int()

        # Freeze sequences that already emitted <eos> so they advance t_src and exit cleanly
        has_eos = (tgt == EOS_IDX).any(dim=1)
        is_blank = (pred[:, -1] == BLANK_IDX) | has_eos
        is_not_blank = ~is_blank

        advance_src = is_blank & ~end_of_src
        advance_tgt = is_not_blank & ~end_of_src

        t_src = torch.where(advance_src, t_src + 1, t_src)
        t_tgt = torch.where(advance_tgt, t_tgt + 1, t_tgt)
        h = torch.where(advance_tgt.view(1, -1, 1), new_h, h)
        c = torch.where(advance_tgt.view(1, -1, 1), new_c, c)

        update_indices = t_tgt.unsqueeze(1)
        update_values = torch.where(
            advance_tgt.unsqueeze(1),
            pred[:, -1].unsqueeze(1),
            tgt.gather(1, update_indices),
        )
        tgt = torch.scatter(tgt, 1, update_indices, update_values)

        hc = torch.stack((h, c), dim=0)
        ts = torch.stack((t_src, t_tgt), dim=1)
        return tgt, hc, ts

    def greedy_decode(
        self,
        src: Tensor,
        src_lengths: Tensor,
        return_early: bool = False,
    ):
        tgt, enc_state, hc, ts = self.encode(src)
        for i in range(self.max_len - 1):
            tgt, hc, ts = self.decode_step(tgt, enc_state, src_lengths, hc, ts)
            if return_early and torch.all(torch.ge(ts[:, 0], src_lengths)):
                break
        return tgt