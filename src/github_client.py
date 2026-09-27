import base64
import os
from datetime import datetime, timezone
from io import BytesIO

import joblib
import requests
import streamlit as st

from src.config import (
    DEFAULT_GITHUB_BRANCH,
    DEFAULT_GITHUB_REPO,
    METRICS_PATH_IN_REPO,
    MODEL_PATH_IN_REPO,
)
from src.storage import make_metrics_json, serialize_bundle


def get_github_config():
    if "GITHUB_TOKEN" not in st.secrets:
        raise RuntimeError("Thiếu GITHUB_TOKEN trong Streamlit Secrets.")

    repo = st.secrets.get("GITHUB_REPO", DEFAULT_GITHUB_REPO)
    branch = st.secrets.get("GITHUB_BRANCH", DEFAULT_GITHUB_BRANCH)

    return {
        "token": st.secrets["GITHUB_TOKEN"],
        "repo": repo,
        "branch": branch,
    }


def github_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "math4ai-streamlit",
    }


def github_contents_url(repo, path):
    clean_path = path.lstrip("/")
    return f"https://api.github.com/repos/{repo}/contents/{clean_path}"


def github_get_file_sha(token, repo, branch, path):
    response = requests.get(
        github_contents_url(repo, path),
        headers=github_headers(token),
        params={"ref": branch},
        timeout=30,
    )

    if response.status_code == 404:
        return None

    if not response.ok:
        raise RuntimeError(
            f"GitHub GET lỗi {response.status_code}: {response.text}"
        )

    return response.json().get("sha")


def github_upsert_bytes(token, repo, branch, path, content, commit_message):
    sha = github_get_file_sha(token, repo, branch, path)

    payload = {
        "message": commit_message,
        "content": base64.b64encode(content).decode("ascii"),
        "branch": branch,
    }

    if sha:
        payload["sha"] = sha

    response = requests.put(
        github_contents_url(repo, path),
        headers=github_headers(token),
        json=payload,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"GitHub PUT lỗi {response.status_code}: {response.text}"
        )

    return response.json()


def github_upsert_text(token, repo, branch, path, text, commit_message):
    return github_upsert_bytes(
        token, repo, branch, path, text.encode("utf-8"), commit_message
    )


def push_bundle_to_github(bundle):
    config = get_github_config()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    model_bytes = serialize_bundle(bundle)
    metrics_text = make_metrics_json(bundle)

    model_result = github_upsert_bytes(
        token=config["token"],
        repo=config["repo"],
        branch=config["branch"],
        path=MODEL_PATH_IN_REPO,
        content=model_bytes,
        commit_message=f"Update trained model - {stamp}",
    )

    metrics_result = github_upsert_text(
        token=config["token"],
        repo=config["repo"],
        branch=config["branch"],
        path=METRICS_PATH_IN_REPO,
        text=metrics_text,
        commit_message=f"Update ML metrics - {stamp}",
    )

    return {
        "repo": config["repo"],
        "branch": config["branch"],
        "model_commit_url": model_result["commit"]["html_url"],
        "metrics_commit_url": metrics_result["commit"]["html_url"],
    }


@st.cache_resource(show_spinner=False)
def load_saved_bundle():
    """Load the saved bundle from the local repo file, falling back to GitHub."""
    if os.path.exists(MODEL_PATH_IN_REPO):
        try:
            return joblib.load(MODEL_PATH_IN_REPO)
        except Exception as e:
            st.warning(f"Tìm thấy model local nhưng load thất bại: {e}")

    try:
        if "GITHUB_TOKEN" not in st.secrets:
            return None

        config = get_github_config()

        response = requests.get(
            github_contents_url(config["repo"], MODEL_PATH_IN_REPO),
            headers=github_headers(config["token"]),
            params={"ref": config["branch"]},
            timeout=60,
        )

        if response.status_code == 404:
            return None

        if not response.ok:
            st.warning(
                f"Không tải được model từ GitHub: {response.status_code}"
            )
            return None

        # GitHub Contents API returns file content base64-encoded
        encoded = response.json().get("content")
        if not encoded:
            return None

        model_bytes = base64.b64decode(encoded)

        return joblib.load(BytesIO(model_bytes))

    except Exception as e:
        st.warning(f"Không thể load model đã lưu từ GitHub: {e}")
        return None