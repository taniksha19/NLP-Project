# src/mask_reg/masker.py
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .hashing import stable_hash_token

_WS = re.compile(r"\s+")
_EXTRA = re.compile(r"<extra_id_\d+>")


@dataclass
class LORArtifact:
    lor_by_hash: Dict[str, float]
    threshold: float
    tokenizer_name: str
    table_path: str


def load_lor_artifact(lang: str, lor_dir: str = "artifacts/lor") -> LORArtifact:
    lor_dir = Path(lor_dir)
    meta_path = lor_dir / f"{lang}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Robust table path resolution
    table_path = Path(meta["table_file"])
    if not table_path.is_absolute():
        candidate = lor_dir / table_path
        if candidate.exists():
            table_path = candidate
    if not table_path.exists():
        table_path = lor_dir / Path(meta["table_file"]).name

    lor_by_hash: Dict[str, float] = {}
    with table_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            lor_by_hash[obj["h"]] = float(obj["lor"])

    return LORArtifact(
        lor_by_hash=lor_by_hash,
        threshold=float(meta["lor_threshold"]),
        tokenizer_name=str(meta.get("tokenizer", "")),
        table_path=str(table_path),
    )


def _normalize_masked(s: str) -> str:
    # Ensure sentinels have spaces around them, then collapse whitespace
    s = _EXTRA.sub(lambda m: f" {m.group(0)} ", s)
    return _WS.sub(" ", s).strip()


def _group_true(flags: List[bool]) -> List[Tuple[int, int]]:
    """Return spans [start, end) over token indices where flags are True."""
    spans: List[Tuple[int, int]] = []
    i, n = 0, len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i + 1
        while j < n and flags[j]:
            j += 1
        spans.append((i, j))
        i = j
    return spans


def _expand_to_whole_word(toks: List[str], spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    SentencePiece heuristic:
    - tokens starting with '▁' typically begin a new word.
    - continuation pieces usually don't start with '▁'.
    Expand each span to include continuation pieces so we mask the whole surface word.
    """
    n = len(toks)
    expanded: List[Tuple[int, int]] = []

    for a, b in spans:
        aa = a
        # Expand left if we started mid-word (continuation piece)
        while aa > 0 and (not toks[aa].startswith("▁")) and toks[aa] != "▁":
            aa -= 1

        bb = b
        # Expand right to include continuation pieces
        while bb < n and (not toks[bb].startswith("▁")) and toks[bb] != "▁":
            bb += 1

        expanded.append((aa, bb))

    # Merge overlaps
    expanded.sort()
    merged: List[List[int]] = []
    for a, b in expanded:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    return [(a, b) for a, b in merged]


def _sentinel_id(tokenizer, i: int) -> int:
    """
    Robustly get the token id for <extra_id_i>.
    Avoids the common failure mode where convert_tokens_to_ids returns unk -> decoded as <unk>.
    """
    # Best case: tokenizer knows sentinel ids
    if hasattr(tokenizer, "get_sentinel_token_ids"):
        ids = tokenizer.get_sentinel_token_ids()
        if i < len(ids):
            return int(ids[i])

    tok = f"<extra_id_{i}>"

    # Try direct conversion
    tid = tokenizer.convert_tokens_to_ids(tok)
    if tid is not None:
        unk = tokenizer.unk_token_id
        if unk is None or tid != unk:
            return int(tid)

    # Fallback: encode literal and expect single id
    ids = tokenizer.encode(tok, add_special_tokens=False)
    if len(ids) == 1:
        unk = tokenizer.unk_token_id
        if unk is None or ids[0] != unk:
            return int(ids[0])

    raise ValueError(
        f"Tokenizer cannot resolve sentinel token id for {tok}. "
        f"Use slow tokenizer (use_fast=False) for T5/mT5."
    )


def mask_text_with_lor(
    text: str,
    tokenizer,
    lor_by_hash: Dict[str, float],
    threshold: float,
    max_spans: int = 10,
    threshold_mult: float = 1.0,
    expand_to_word: bool = True,
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Token-level masking (no offset_mapping needed).
    Returns:
      masked_text, spans over token indices [start, end)
    """
    enc = tokenizer(text, add_special_tokens=False)
    input_ids: List[int] = enc["input_ids"]
    toks: List[str] = tokenizer.convert_ids_to_tokens(input_ids)

    eff_thr = float(threshold) * float(threshold_mult)

    flags: List[bool] = []
    for t in toks:
        h = stable_hash_token(t)
        s = float(lor_by_hash.get(h, 0.0))
        flags.append(s >= eff_thr)

    spans = _group_true(flags)
    if not spans:
        return text, []

    if expand_to_word:
        spans = _expand_to_whole_word(toks, spans)

    spans = spans[:max_spans]

    # Replace each toxic span with <extra_id_i> at the token level
    sentinel_ids = [_sentinel_id(tokenizer, i) for i in range(len(spans))]

    new_ids: List[int] = []
    cur = 0
    for i, (a, b) in enumerate(spans):
        new_ids.extend(input_ids[cur:a])
        new_ids.append(sentinel_ids[i])
        cur = b
    new_ids.extend(input_ids[cur:])

    masked = tokenizer.decode(new_ids, skip_special_tokens=False)
    masked = _normalize_masked(masked)

    # Guardrail: if sentinel got broken, fail loudly instead of silently outputting <unk>
    if "<unk>" in masked:
        raise RuntimeError(
            "Masking produced <unk>. This usually means your tokenizer can't resolve <extra_id_*>. "
            "Load tokenizer with use_fast=False (slow tokenizer) in both train and infer."
        )

    return masked, spans
