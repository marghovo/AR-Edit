"""Prompt alignment: which words differ between the source and target prompts.

The token sequences of the two prompts are aligned with Needleman-Wunsch
(`utils/seq_aligner.py`, from ReFlex's aligner - https://arxiv.org/pdf/2507.01496,
                     https://github.com/wlaud1001/ReFlex).
                     From the alignment we derive three sets of
                     text-token positions that drive the edit:

* target-only ("gap") tokens  -> boosted in cross-attention, edit-mask source
* removed source tokens       -> edit-mask source (where the old content lives)
* kept source tokens          -> what the anchor must preserve

Function words and special tokens are dropped from the three sets: their
cross-attention columns are diffuse and only blur the saliency maps.
"""

from __future__ import annotations

import torch
from typing import List
from dataclasses import dataclass, field
from utils.seq_aligner import get_text_alphas

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "on", "in", "at", "by", "to", "with",
    "under", "over", "past", "is", "are", "was", "were", "its", "it", "as",
    "for", "from", "this", "that", "these", "those", "through", "across",
    "into", "onto", "while", "during", "up", "down", "out", "off", "he",
    "she", "him", "her", "his", "their", "they", "s", "t",
})
SPECIAL_TOKENS = frozenset({"</s>", "<pad>", "<unk>"})


@dataclass
class PromptAlignment:
    """Result of `align_prompts`."""

    tar_alphas: torch.Tensor                      # [max_len]: 1 = aligned, 0 = target-only
    tar_gap_idxs: List[int] = field(default_factory=list)
    src_removed_idxs: List[int] = field(default_factory=list)
    src_kept_idxs: List[int] = field(default_factory=list)

    @property
    def num_gap_tokens(self) -> int:
        return int((self.tar_alphas == 0).sum().item())

    @property
    def has_edit_tokens(self) -> bool:
        """False when the two prompts are (token-wise) identical: no anchoring possible."""
        return bool(self.tar_gap_idxs or self.src_removed_idxs)


def content_token_idxs(tokenizer, prompt: str, idxs: List[int]) -> List[int]:
    """Keep only positions of content words (drop stop words, special tokens,
    punctuation). `tokenizer` is the UMT5 HuggingfaceTokenizer wrapper."""
    ids = tokenizer(prompt, padding=False, truncation=False)[0].tolist()
    toks = tokenizer.tokenizer.convert_ids_to_tokens(ids)
    keep = []
    for j in idxs:
        if j >= len(toks):
            continue
        raw = toks[j]
        if raw in SPECIAL_TOKENS:
            continue
        t = raw.replace("▁", "").strip().lower()
        if not t or not any(ch.isalnum() for ch in t):
            continue
        if t in STOPWORDS:
            continue
        keep.append(j)
    return keep


def align_prompts(tokenizer, src_prompt: str, tar_prompt: str, max_len: int = 512) -> PromptAlignment:
    """Align the two prompts and return the token sets that drive the edit."""
    tar_alphas = get_text_alphas(src_prompt, tar_prompt, tokenizer, max_len=max_len)
    src_alphas = get_text_alphas(tar_prompt, src_prompt, tokenizer, max_len=max_len)
    tar_gap_idxs = (tar_alphas == 0).nonzero(as_tuple=True)[0].tolist()
    src_removed_idxs = (src_alphas == 0).nonzero(as_tuple=True)[0].tolist()
    n_src_tok = len(tokenizer(src_prompt, padding=False, truncation=False)[0])
    src_kept_idxs = [j for j in range(n_src_tok) if src_alphas[j] >= 0.5]
    return PromptAlignment(
        tar_alphas=tar_alphas,
        tar_gap_idxs=content_token_idxs(tokenizer, tar_prompt, tar_gap_idxs),
        src_removed_idxs=content_token_idxs(tokenizer, src_prompt, src_removed_idxs),
        src_kept_idxs=content_token_idxs(tokenizer, src_prompt, src_kept_idxs),
    )
