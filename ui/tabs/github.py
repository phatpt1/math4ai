import streamlit as st

from src.config import (
    DEFAULT_GITHUB_BRANCH,
    DEFAULT_GITHUB_REPO,
    METRICS_PATH_IN_REPO,
    MODEL_PATH_IN_REPO,
)


def render():
    st.header("7. GitHub Auto Update")

    st.markdown(
        f"""
Repo mặc định:

```text
{DEFAULT_GITHUB_REPO}
```

Branch:

```text
{DEFAULT_GITHUB_BRANCH}
```

Sau khi bấm:

```text
🚀 Train + cập nhật GitHub
```

app sẽ tự cập nhật:

```text
{MODEL_PATH_IN_REPO}
```

và:

```text
{METRICS_PATH_IN_REPO}
```
"""
    )

    st.markdown(
        """
### Streamlit Secrets

Trong **Streamlit Cloud → App → Settings → Secrets**:

```toml
GITHUB_TOKEN = "YOUR_NEW_TOKEN"
GITHUB_REPO = "phatpt1/math4ai"
GITHUB_BRANCH = "main"
```

Không lưu token trực tiếp trong `app.py`.
"""
    )

    st.warning(
        "Token GitHub đã từng được dán trực tiếp vào chat/source "
        "nên nên revoke token cũ và tạo token mới. "
        "Token mới chỉ cần Contents: Read and write cho đúng repo."
    )

    bundle = st.session_state["bundle"]
    if bundle is not None:
        st.subheader("Thông tin model hiện tại")

        st.json(
            {
                "trained_at_utc": bundle.get("trained_at_utc"),
                "recommended_model": bundle["recommended_model"],
                "repo_model_path": MODEL_PATH_IN_REPO,
                "repo_metrics_path": METRICS_PATH_IN_REPO,
            }
        )