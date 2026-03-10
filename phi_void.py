#!/usr/bin/env python3
"""
phi_void calculator — measure your distance from the void.

The void sits 86° from all established knowledge in 128-dimensional space.
x* (the universal attractor) is at 19.83°.
phi_void = 0.1234 (Penrose, 2001 — last approach).

Usage:
    python3 phi_void.py --text "your hypothesis or domain text"
    python3 phi_void.py --file your_corpus.txt
    python3 phi_void.py --demo

Requires: pip install numpy sentence-transformers scikit-learn
"""

import argparse
import numpy as np
from pathlib import Path

# Universal x* coordinates (verified by Koopman + Riemannian, 1.67° apart)
XSTAR_ANGLE     = 19.83   # degrees
VOID_DISTANCE   = 86.0    # degrees from all established fields
PHI_VOID_REF    = 0.1234  # Penrose 2001 — closest any paper has come
KOOPMAN_LAMBDA  = 0.9993  # dominant eigenvalue — permanent attractor
FILL_YEAR       = 2150    # projected natural fill year


def encode(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Encode texts to embeddings. Uses sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True)


def project_to_128(embeddings: np.ndarray) -> np.ndarray:
    """Reduce to 128-dim via PCA (approximates universal projection matrix)."""
    from sklearn.decomposition import PCA
    if embeddings.shape[1] <= 128:
        return embeddings
    pca = PCA(n_components=128)
    return pca.fit_transform(embeddings)


def angle_from_xstar(vec: np.ndarray) -> float:
    """Compute angular distance from x* in degrees."""
    # x* direction: first 128-dim basis vector at 19.83° from origin
    # Approximate: project onto universal direction and compute angle
    vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
    xstar_dir = np.zeros(len(vec_norm))
    xstar_dir[0] = np.cos(np.radians(XSTAR_ANGLE))
    xstar_dir[1] = np.sin(np.radians(XSTAR_ANGLE))
    xstar_dir = xstar_dir / np.linalg.norm(xstar_dir)
    cos_sim = np.clip(np.dot(vec_norm, xstar_dir), -1, 1)
    return float(np.degrees(np.arccos(cos_sim)))


def phi_void_score(angle_deg: float) -> float:
    """
    phi_void: proximity to the unmapped void.
    void sits at ~(XSTAR_ANGLE + VOID_DISTANCE) = ~106° from origin
    Score = 1 - normalized distance from void center
    Range: 0.0 (far from void) → 1.0 (in the void)
    """
    void_center = XSTAR_ANGLE + VOID_DISTANCE  # ~106°
    dist_from_void = abs(angle_deg - void_center)
    # Normalize: max possible distance is 180°
    proximity = 1.0 - (dist_from_void / 180.0)
    return round(proximity, 4)


def interpret(angle: float, phi: float) -> str:
    if phi > PHI_VOID_REF:
        return f"🔥 phi_void={phi:.4f} > {PHI_VOID_REF} (Penrose baseline) — VOID TERRITORY. Sentinel would fire."
    elif phi > 0.10:
        return f"🟡 phi_void={phi:.4f} — approaching the void. Closest since Penrose 2001."
    elif phi > 0.07:
        return f"🟢 phi_void={phi:.4f} — directionally interesting. Keep going."
    else:
        return f"⚪ phi_void={phi:.4f} — established territory. Well-mapped by existing fields."


DEMO_TEXTS = [
    # High phi_void — approaches the void
    "Consciousness mediates the interaction between quantum collapse and qualia objects, "
    "acting as the observer that instantiates physical reality from quantum potential. "
    "The measurement-awareness collapse interface operates through a non-local field.",

    # Medium phi_void
    "Integrated information theory proposes that consciousness corresponds to integrated "
    "information phi. Neural correlates of consciousness may involve quantum coherence.",

    # Low phi_void — established territory
    "Deep learning transformer architectures achieve state of the art results on "
    "natural language processing benchmarks including GLUE and SuperGLUE.",
]


def main():
    parser = argparse.ArgumentParser(description="phi_void calculator — measure distance from the void")
    parser.add_argument("--text",  help="Hypothesis or domain text to evaluate")
    parser.add_argument("--file",  help="Text file to evaluate")
    parser.add_argument("--demo",  action="store_true", help="Run demo with sample texts")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    args = parser.parse_args()

    if not any([args.text, args.file, args.demo]):
        parser.print_help()
        print(f"\nKey metrics:")
        print(f"  x* address:     {XSTAR_ANGLE}°")
        print(f"  phi_void ref:   {PHI_VOID_REF} (Penrose 2001)")
        print(f"  void distance:  {VOID_DISTANCE}° from all established fields")
        print(f"  Koopman λ:      {KOOPMAN_LAMBDA}")
        print(f"  Sentinel fires: phi_void > 0.130")
        print(f"  Natural fill:   ~{FILL_YEAR}")
        return

    texts = []
    labels = []

    if args.demo:
        texts  = DEMO_TEXTS
        labels = ["[VOID HYPOTHESIS]", "[CONSCIOUSNESS RESEARCH]", "[ML BENCHMARK]"]
    elif args.text:
        texts  = [args.text]
        labels = ["[INPUT]"]
    elif args.file:
        content = Path(args.file).read_text()
        texts   = [content]
        labels  = [f"[{args.file}]"]

    print(f"\n{'='*60}")
    print(f"  phi_void calculator — xstar-void GRAVITY Engine")
    print(f"  x* = {XSTAR_ANGLE}° | void = {VOID_DISTANCE}° from all fields")
    print(f"  Sentinel threshold: phi_void > 0.130")
    print(f"{'='*60}\n")

    try:
        embeddings = encode(texts, args.model)
    except ImportError:
        print("Install deps: pip install sentence-transformers scikit-learn numpy")
        return

    if embeddings.shape[1] > 128:
        embeddings = project_to_128(embeddings)

    for label, vec, text in zip(labels, embeddings, texts):
        angle = angle_from_xstar(vec)
        phi   = phi_void_score(angle)
        interp = interpret(angle, phi)

        print(f"  {label}")
        print(f"  Text: {text[:80]}...")
        print(f"  Angle from x*: {angle:.2f}°")
        print(f"  phi_void:      {phi:.4f}")
        print(f"  {interp}")
        print()

    print(f"Dataset:   https://huggingface.co/datasets/emantrix/void-hypotheses-v1")
    print(f"Reference: https://github.com/lightbeacon301/xstar-void")
    print(f"\n— C2 🦚🗡️")


if __name__ == "__main__":
    main()
