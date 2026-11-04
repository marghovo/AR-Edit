"""Token alignment via Needleman-Wunsch.

From https://github.com/wlaud1001/ReFlex and
https://github.com/google/prompt-to-prompt/ (seq_aligner).

"""

import torch


class _Score:
    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, a, b):
        return self.match if a == b else self.mismatch


def _global_align(x, y, score):
    M, N = len(x), len(y)
    matrix = [[0] * (N + 1) for _ in range(M + 1)]
    trace = [[0] * (N + 1) for _ in range(M + 1)]  # 1=diag 2=up 3=left
    for i in range(1, M + 1):
        matrix[i][0] = matrix[i - 1][0] + score.gap
        trace[i][0] = 2
    for j in range(1, N + 1):
        matrix[0][j] = matrix[0][j - 1] + score.gap
        trace[0][j] = 3
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            diag = matrix[i - 1][j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            up   = matrix[i - 1][j] + score.gap
            left = matrix[i][j - 1] + score.gap
            best = max(diag, up, left)
            matrix[i][j] = best
            trace[i][j] = 1 if best == diag else (2 if best == up else 3)
    return matrix, trace


def _traceback(x, y, trace):
    i, j = len(x), len(y)
    pairs = []
    while i > 0 or j > 0:
        t = trace[i][j]
        if t == 1:
            pairs.append((j - 1, i - 1))  # (target_idx, source_idx)
            i -= 1; j -= 1
        elif t == 2:
            i -= 1
        else:
            pairs.append((j - 1, -1))     # target token has no source counterpart (gap)
            j -= 1
    pairs.reverse()
    return torch.tensor(pairs, dtype=torch.long) if pairs else torch.zeros((0, 2), dtype=torch.long)


def _to_1d_int_list(out):
    """Normalize a tokenizer's output to a 1D Python list of int token IDs."""
    if isinstance(out, dict):
        ids = out.get("input_ids", out.get("ids"))
    else:
        ids = out
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    # Flatten one level of batching if present.
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def get_text_alphas(source_prompt: str, target_prompt: str, tokenizer, max_len: int = 512) -> torch.Tensor:
    """Return alphas[max_len] float tensor: 1.0 for matched target token, 0.0 for gap.

    Padding positions beyond the alignment length stay at 1.0 (pass-through).
    """
    if not source_prompt:
        return torch.ones(max_len, dtype=torch.float32)

    if hasattr(tokenizer, "encode"):
        x_raw = tokenizer.encode(source_prompt)
        y_raw = tokenizer.encode(target_prompt)
    else:
        x_raw = tokenizer(source_prompt)
        y_raw = tokenizer(target_prompt)
    x_seq = _to_1d_int_list(x_raw)
    y_seq = _to_1d_int_list(y_raw)

    score = _Score(gap=0, match=1, mismatch=-1)
    _, trace = _global_align(x_seq, y_seq, score)
    aligned = _traceback(x_seq, y_seq, trace)  # [P, 2]

    alphas = torch.ones(max_len, dtype=torch.float32)
    if aligned.numel() > 0:
        # mark only the aligned-target-position slots; gap rows get 0.0
        n_pairs = aligned.shape[0]
        alphas[:n_pairs] = aligned[:, 1].ne(-1).float()
    return alphas
