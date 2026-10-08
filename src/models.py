"""Pan-allele CNN and a small transformer on BLOSUM and one-hot encodings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.constants import BLOSUM62, BLOSUM_SCALE, N_AA, PAD_IDX, UNK_IDX


def _aa_features(idx: torch.Tensor, blosum: torch.Tensor, one_hot: bool) -> torch.Tensor:
    """Map indices to scaled BLOSUM rows, optionally concatenated with one-hot.

    PAD and UNK positions are zeros. ``idx`` is (batch, length).
    """
    known = (idx != PAD_IDX) & (idx != UNK_IDX)
    safe = idx.clamp(0, N_AA - 1)
    features = blosum[safe] / BLOSUM_SCALE
    features = features * known.unsqueeze(-1)
    if one_hot:
        eye = F.one_hot(safe, N_AA).to(dtype=features.dtype)
        eye = eye * known.unsqueeze(-1)
        features = torch.cat([features, eye], dim=-1)
    return features


def _masked_max(hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Max over length. ``hidden`` is (batch, channels, length), ``valid`` is (batch, length)."""
    filled = hidden.masked_fill(~valid.unsqueeze(1), -1e4)
    return filled.max(dim=-1).values


class PanAlleleCNN(nn.Module):
    """Early-fusion CNN.

    The MHC pseudo-sequence is convolved and pooled, tiled onto every peptide
    position, and mixed with the peptide's BLOSUM+one-hot features.
    P2 and the C-terminus are also concatenated after pooling, because those
    are the MHC class I anchor positions and left-alignment alone does not
    put the C-terminus in a fixed column when lengths differ.
    """

    def __init__(self, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.register_buffer("blosum", torch.tensor(BLOSUM62, dtype=torch.float32))
        self.mhc_conv = nn.Sequential(
            nn.Conv1d(N_AA, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pep_conv = nn.Sequential(
            nn.Conv1d(2 * N_AA + hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 4 * N_AA, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, pep_idx: torch.Tensor, mhc_idx: torch.Tensor) -> torch.Tensor:
        pep_valid = pep_idx != PAD_IDX
        mhc_valid = mhc_idx != PAD_IDX
        pep = _aa_features(pep_idx, self.blosum, one_hot=True)
        mhc = _aa_features(mhc_idx, self.blosum, one_hot=False)
        mhc_h = self.mhc_conv(mhc.transpose(1, 2))
        mhc_pool = _masked_max(mhc_h, mhc_valid)
        tiled = mhc_pool.unsqueeze(1).expand(-1, pep.size(1), -1)
        fused = torch.cat([pep, tiled], dim=-1)
        pep_h = self.pep_conv(fused.transpose(1, 2))
        pooled = _masked_max(pep_h, pep_valid)
        anchors = torch.cat(
            [_anchor_features(pep_idx, self.blosum, which="p2"), _anchor_features(pep_idx, self.blosum, which="cterm")],
            dim=-1,
        )
        logits = self.head(torch.cat([pooled, anchors], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)


class SmallTransformer(nn.Module):
    """Encoder over peptide tokens then MHC tokens, with a CLS output.

    Peptide tokens get a position id from the N-terminus and another from
    the C-terminus, so the C-terminal anchor is visible at every length.
    """

    def __init__(
        self,
        d_model: int = 48,
        nhead: int = 4,
        nlayers: int = 2,
        dim_feedforward: int = 96,
        dropout: float = 0.1,
        max_pep_len: int = 15,
        max_mhc_len: int = 34,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}.")
        self.max_pep_len = max_pep_len
        self.register_buffer("blosum", torch.tensor(BLOSUM62, dtype=torch.float32))
        self.pep_proj = nn.Linear(2 * N_AA, d_model)
        self.mhc_proj = nn.Linear(N_AA, d_model)
        self.pos_n = nn.Embedding(max_pep_len, d_model)
        self.pos_c = nn.Embedding(max_pep_len, d_model)
        self.pos_mhc = nn.Embedding(max_mhc_len, d_model)
        self.type_emb = nn.Embedding(2, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers, enable_nested_tensor=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, pep_idx: torch.Tensor, mhc_idx: torch.Tensor) -> torch.Tensor:
        batch, pep_len = pep_idx.shape
        mhc_len = mhc_idx.size(1)
        pep_valid = pep_idx != PAD_IDX
        mhc_valid = mhc_idx != PAD_IDX
        pep = self.pep_proj(_aa_features(pep_idx, self.blosum, one_hot=True))
        mhc = self.mhc_proj(_aa_features(mhc_idx, self.blosum, one_hot=False))
        n_pos = torch.arange(pep_len, device=pep_idx.device).unsqueeze(0).expand(batch, -1)
        lengths = pep_valid.sum(dim=1).clamp(min=1)
        c_pos = (lengths.unsqueeze(1) - 1 - n_pos).clamp(0, self.max_pep_len - 1)
        pep = pep + self.pos_n(n_pos) + self.pos_c(c_pos) + self.type_emb.weight[0]
        m_pos = torch.arange(mhc_len, device=mhc_idx.device).unsqueeze(0).expand(batch, -1)
        mhc = mhc + self.pos_mhc(m_pos) + self.type_emb.weight[1]
        cls = self.cls.expand(batch, -1, -1)
        tokens = torch.cat([cls, pep, mhc], dim=1)
        keep_cls = torch.zeros(batch, 1, dtype=torch.bool, device=pep_idx.device)
        pad_mask = torch.cat([keep_cls, ~pep_valid, ~mhc_valid], dim=1)
        hidden = self.encoder(tokens, src_key_padding_mask=pad_mask)
        logits = self.head(hidden[:, 0, :]).squeeze(-1)
        return torch.sigmoid(logits)


def _anchor_features(pep_idx: torch.Tensor, blosum: torch.Tensor, which: str) -> torch.Tensor:
    """BLOSUM+one-hot of P2 (index 1) or the last non-pad residue."""
    valid = pep_idx != PAD_IDX
    if which == "p2":
        chosen = pep_idx[:, 1] if pep_idx.size(1) > 1 else pep_idx[:, 0]
        present = valid[:, 1] if pep_idx.size(1) > 1 else valid[:, 0]
    else:
        lengths = valid.sum(dim=1).clamp(min=1)
        last = (lengths - 1).unsqueeze(1)
        chosen = pep_idx.gather(1, last).squeeze(1)
        present = lengths > 0
    features = _aa_features(chosen.unsqueeze(1), blosum, one_hot=True).squeeze(1)
    return features * present.unsqueeze(-1).to(features.dtype)
