"""Task 1: follow the freznelai organization and download its two models, using only the HF Python API."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, constants, get_session, hf_raise_for_status, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import build_hf_headers

ORG = "freznelai"
MODELS = [
    "freznelai/FreznelAI_1.0_Face-Detector_500M_FZFP4_FRZm",
    "freznelai/FreznelAI_1.0_Face-Landmarker_500M_FZFP4_FRZm",
]
MODELS_DIR = Path("models")


def follow_org(api: HfApi, token: str, org: str) -> None:
    # huggingface_hub has no follow method; this is the Hub endpoint the website's Follow button uses.
    resp = get_session().post(
        f"{constants.ENDPOINT}/api/organizations/{org}/follow",
        headers=build_hf_headers(token=token),
    )
    hf_raise_for_status(resp)
    if not api.get_organization_overview(org, token=token).is_following:
        raise RuntimeError(f"Follow request succeeded but Hub reports is_following=False for {org}")


def download_and_verify(api: HfApi, token: str, repo_id: str) -> bool:
    local_dir = MODELS_DIR / repo_id.split("/")[1]
    snapshot_download(repo_id, local_dir=local_dir, token=token)
    ok = True
    for sib in api.model_info(repo_id, files_metadata=True, token=token).siblings:
        path = local_dir / sib.rfilename
        size = path.stat().st_size if path.exists() else None
        status = "OK" if size == sib.size else "MISMATCH"
        ok &= status == "OK"
        print(f"  {status:8} {sib.rfilename:50} hub={sib.size} local={size}")
    return ok


def main() -> int:
    load_dotenv()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is not set. Copy .env.example to .env and add your Hugging Face token.")
        return 1
    api = HfApi(token=token)
    print(f"Logged in as: {api.whoami()['name']}")

    try:
        follow_org(api, token, ORG)
    except (HfHubHTTPError, RuntimeError) as e:
        print(f"FAILED to follow {ORG}: {e}")
        return 1
    print(f"Following {ORG}: is_following=True")

    all_ok = True
    for repo_id in MODELS:
        print(f"Downloading {repo_id} -> {MODELS_DIR / repo_id.split('/')[1]}")
        all_ok &= download_and_verify(api, token, repo_id)
    print("All model files verified." if all_ok else "Some files did not match the Hub.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
