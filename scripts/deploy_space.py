"""Mirror the API image sources to a Hugging Face Space (Docker SDK).

Run by .github/workflows/space.yml with HF_TOKEN; HF builds the same Dockerfile.

    HF_TOKEN=... uv run --no-project --with huggingface_hub python scripts/deploy_space.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

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


def fail(msg: str) -> NoReturn:
    # "::error::" becomes an annotation on the Actions run page, visible without signing in
    print(f"::error::{msg}")
    sys.exit(1)


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        fail("HF_TOKEN is empty: add it under Settings -> Secrets and variables -> Actions "
             "-> Repository secrets (not an environment secret)")  # fmt: skip
    api = HfApi(token=token)
    try:
        who = api.whoami()
    except Exception as e:  # noqa: BLE001 - report any auth failure the same way
        fail(f"HF_TOKEN was rejected by Hugging Face ({type(e).__name__}); create a new token")
    role = who.get("auth", {}).get("accessToken", {}).get("role", "?")
    owner = SPACE.split("/")[0]
    print(f"authenticated as {who.get('name')} (token role: {role})")
    if who.get("name") != owner:
        fail(f"token belongs to {who.get('name')!r}, but the Space owner is {owner!r}")
    if role == "read":
        fail("HF_TOKEN is read-only: create a token with Write access")
    try:
        if not api.repo_exists(SPACE, repo_type="space"):  # creating may be refused (402)
            api.create_repo(SPACE, repo_type="space", space_sdk="docker")
    except Exception as e:  # noqa: BLE001
        detail = getattr(e, "server_message", None) or str(e)
        hint = ""
        if "402" in str(e):
            hint = (" The account may not create Spaces through the API (e.g. unverified email);"
                    f" create https://huggingface.co/spaces/{SPACE} once in the web UI"
                    " (SDK: Docker, Blank, CPU basic), then re-run this workflow.")  # fmt: skip
        fail(f"could not create {SPACE}: {detail[:300]}.{hint}")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for f in FILES:
            shutil.copy2(ROOT / f, stage / f)
        for d in DIRS:
            shutil.copytree(ROOT / d, stage / d, ignore=shutil.ignore_patterns("__pycache__"))
        (stage / "README.md").write_text(SPACE_README)  # also the package readme in the image
        sha = os.environ.get("GITHUB_SHA", "local")[:7]
        try:
            api.upload_folder(
                folder_path=stage,
                repo_id=SPACE,
                repo_type="space",
                commit_message=f"Deploy {sha}",
                delete_patterns=["*"],  # mirror: drop files that are gone from the source
            )
        except Exception as e:  # noqa: BLE001
            fail(f"upload to {SPACE} failed: {type(e).__name__}: {str(e)[:200]}")
    print(f"deployed to https://huggingface.co/spaces/{SPACE}")


if __name__ == "__main__":
    main()
