"""Suggest FA_MATCH_THRESHOLD for the active embedder from the enrollment crops in SQLite.

Leave-one-out, matching what attendance does (best score over a person's pose vectors):
- genuine: each crop vs the same person's other poses
- impostor: each crop vs every other person
    uv run python scripts/calibrate_threshold.py
Needs >= 2 enrolled people; more people = more trustworthy numbers.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.embedder import Embedder  # noqa: E402
from core.repository import Repository  # noqa: E402
from utils.config import get_settings  # noqa: E402


def scores(by_person: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """by_person: person_id -> (P, 512) normalized embeddings -> (genuine, impostor) scores."""
    genuine, impostor = [], []
    for pid, embs in by_person.items():
        for i, e in enumerate(embs):
            others = np.delete(embs, i, axis=0)
            if len(others):
                genuine.append(float((others @ e).max()))
            impostor += [float((v @ e).max()) for q, v in by_person.items() if q != pid]
    return np.array(genuine), np.array(impostor)


def main() -> None:
    s = get_settings()
    crops: dict[int, list] = defaultdict(list)
    for pid, _, _, _, img in Repository(s.db_path).person_faces():
        crops[pid].append(img)
    if len(crops) < 2:
        raise SystemExit("Cần ít nhất 2 người đã đăng ký (có ảnh lưu sẵn)")
    emb = Embedder(s)
    genuine, impostor = scores({pid: emb.embed_batch(c) for pid, c in crops.items()})
    n_people, n_gen, n_imp = len(crops), len(genuine), len(impostor)
    print(f"{s.embedder_model}: {n_people} người, {n_gen} genuine, {n_imp} impostor")
    print(f"impostor max {impostor.max():.3f}   genuine min {genuine.min():.3f}\n")
    print("threshold   FAR (nhận nhầm)   FRR (bỏ sót)")
    for t in np.arange(0.20, 0.71, 0.05):
        print(f"  {t:.2f}        {np.mean(impostor >= t):6.2%}         {np.mean(genuine < t):6.2%}")
    # Above every impostor seen, with a margin since a small gallery underestimates the tail.
    suggested = max(impostor.max() + 0.1, 0.3)
    print(f"\nĐề xuất FA_MATCH_THRESHOLD={suggested:.2f} (hiện tại {s.match_threshold})")


if __name__ == "__main__":
    main()
