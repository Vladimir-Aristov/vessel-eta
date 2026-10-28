"""Mirror the API image sources to a Hugging Face Space (Docker SDK).

Run by .github/workflows/space.yml with HF_TOKEN; HF builds the same Dockerfile.

    HF_TOKEN=... uv run --no-project --with huggingface_hub python scripts/deploy_space.py
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE = os.environ.get("HF_SPACE", "pchelka1/vessel-eta")
FILES = ["Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "LICENSE"]
DIRS = ["src", "configs", "artifacts"]

SPACE_README = """---
title: vessel-eta
emoji: 🚢
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Probabilistic ship ETA to Los Angeles / Long Beach from AIS
---

# vessel-eta — live demo

Probabilistic ETA for ships approaching the ports of Los Angeles and Long Beach,
from NOAA AIS data. The map replays real December 2024 voyages the model never saw;
the API is at `/docs`.

Source, data pipeline and evaluation: https://github.com/Vladimir-Aristov/vessel-eta
"""


def main() -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(SPACE, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for f in FILES:
            shutil.copy2(ROOT / f, stage / f)
        for d in DIRS:
            shutil.copytree(ROOT / d, stage / d, ignore=shutil.ignore_patterns("__pycache__"))
        (stage / "README.md").write_text(SPACE_README)  # also the package readme in the image
        sha = os.environ.get("GITHUB_SHA", "local")[:7]
        api.upload_folder(
            folder_path=stage,
            repo_id=SPACE,
            repo_type="space",
            commit_message=f"Deploy {sha}",
            delete_patterns=["*"],  # mirror: drop files that are gone from the source
        )
    print(f"deployed to https://huggingface.co/spaces/{SPACE}")


if __name__ == "__main__":
    main()
