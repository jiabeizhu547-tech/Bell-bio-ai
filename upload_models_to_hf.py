"""
Upload custom model weights to HuggingFace Hub model repo.

Usage:
    python upload_models_to_hf.py

This uploads best_model.pt and best_model_esm2.pt to:
    https://huggingface.co/jiabeizhu547-tech/protein-ai-models

Requirements:
    - pip install huggingface_hub
    - A HuggingFace account with write access
    - Login first: huggingface-cli login
      Or set env var: HF_TOKEN=your_token_here
"""

import os
from huggingface_hub import HfApi, create_repo, upload_file

REPO_ID = "jiabeizhu547-tech/protein-ai-models"
INFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "protein-ai-web", "inference")

FILES = ["best_model.pt", "best_model_esm2.pt"]


def main():
    api = HfApi()

    # Create repo (or connect to existing)
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
        print(f"[OK] Repo ready: https://huggingface.co/{REPO_ID}")
    except Exception:
        print(f"[--] Repo already exists or manual creation needed: {REPO_ID}")

    for filename in FILES:
        filepath = os.path.join(INFERENCE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[SKIP] {filename} — file not found at {filepath}")
            continue

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[UP] {filename} ({size_mb:.1f} MB) → {REPO_ID} ...", end=" ", flush=True)

        upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print("OK")

    print("\n[DONE] Models uploaded!")
    print(f"  View: https://huggingface.co/{REPO_ID}")
    print(f"  Now the Dockerfile can download them during HF Spaces build.")


if __name__ == "__main__":
    main()
