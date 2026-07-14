"""
Setup script — downloads model weights required for Protein AI Web.

Run once before starting the server:
    python setup.py
"""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
INFERENCE_DIR = BASE_DIR / "inference"

def main():
    missing = []

    # 1. ESM-2 base model (facebook/esm2_t6_8M_UR50D)
    esm_dir = INFERENCE_DIR / "esm_model"
    if not (esm_dir / "model.safetensors").exists():
        print("[1/3] Downloading ESM-2 model from HuggingFace...")
        try:
            from transformers import EsmForMaskedLM
            EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
            print("  ✓ ESM-2 model cached by HuggingFace")
        except Exception as e:
            print(f"  ⚠ HF download failed: {e}")
            print("  → Manual: download facebook/esm2_t6_8M_UR50D to inference/esm_model/")
    else:
        print("[1/3] ESM-2 model: ✓ found locally")

    # 2. Custom-trained model checkpoints
    for fname, desc in [
        ("best_model.pt", "V1 CNN+BiLSTM (Q3 89.8%)"),
        ("best_model_esm2.pt", "ESM-2 fine-tuned"),
    ]:
        if (INFERENCE_DIR / fname).exists():
            print(f"[ ] {fname}: ✓ found")
        else:
            missing.append(fname)
            print(f"[ ] {fname}: ✗ missing ({desc})")

    if missing:
        # Try to copy from parent project
        parent_candidates = [
            Path(__file__).parent.parent,  # repo root (bell-bio-ai-repo)
            Path("d:/code_test/protein-ai/hf_space"),
        ]

        for src_dir in parent_candidates:
            if not src_dir.exists():
                continue
            for fname in missing[:]:
                src = src_dir / fname
                dst = INFERENCE_DIR / fname
                if src.exists() and not dst.exists():
                    import shutil
                    shutil.copy2(src, dst)
                    print(f"  → copied {fname} from {src_dir}")
                    missing.remove(fname)

    if missing:
        print(f"\n⚠ Missing model files: {missing}")
        print("  These are custom-trained checkpoints not hosted publicly.")
        print("  Place them in inference/ directory to enable SS prediction.")
        print("  EC and Mutation predictions only need the ESM base model.")

    print("\nDone. Run 'python server.py' to start.")

if __name__ == "__main__":
    main()
