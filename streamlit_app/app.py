"""
Streamlit frontend for the crack/no_crack classifier.

This app does NOT load the TensorFlow model itself -- it calls the deployed
FastAPI backend (running on Render, containerized with Docker) over HTTP.
That split -- lightweight UI talking to a separately-deployed model API --
is the standard production pattern for serving ML models, as opposed to
bundling the model directly into the UI process (which is what Project 1's
Streamlit app does, for comparison).

Run locally:
    streamlit run streamlit_app/app.py
"""

import io

import requests
import streamlit as st
from PIL import Image

DEFAULT_API_URL = "https://crack-detect-classiffier.onrender.com"


def main():
    st.set_page_config(page_title="Crack Defect Classifier", layout="centered")
    st.title("Surface Crack Defect Classifier")
    st.caption(
        "CNN served via a separate FastAPI + Docker backend. "
        "This UI just uploads an image and displays the API's response."
    )

    st.sidebar.header("Backend")
    api_url = st.sidebar.text_input("API base URL", value=DEFAULT_API_URL).rstrip("/")

    if st.sidebar.button("Check API health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=15)
            if r.status_code == 200:
                st.sidebar.success(f"API is up: {r.json()}")
            else:
                st.sidebar.error(f"API returned status {r.status_code}")
        except requests.RequestException as e:
            st.sidebar.error(f"Could not reach API: {e}")
    st.sidebar.caption(
        "Note: the free Render tier sleeps after inactivity -- the first "
        "request after idle time can take ~30s while it wakes up."
    )

    uploaded = st.file_uploader("Upload a surface image (crack or no_crack)", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("Upload an image to run detection.")
        return

    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", width=300)

    if st.button("Run detection", type="primary"):
        with st.spinner("Calling the model API..."):
            try:
                uploaded.seek(0)
                files = {"file": (uploaded.name, uploaded.read(), uploaded.type)}
                r = requests.post(f"{api_url}/predict", files=files, timeout=60)
            except requests.RequestException as e:
                st.error(f"Request to API failed: {e}")
                return

        if r.status_code != 200:
            st.error(f"API returned an error ({r.status_code}): {r.text}")
            return

        result = r.json()
        predicted = result["predicted_class"]
        confidence = result["confidence"]

        if predicted == "crack":
            st.error(f"**Crack detected** — confidence {confidence:.1%}")
        else:
            st.success(f"**No crack detected** — confidence {confidence:.1%}")

        st.progress(confidence)
        st.json(result)

    st.divider()
    st.subheader("Recent predictions (from backend's SQL log)")
    try:
        r = requests.get(f"{api_url}/logs?limit=10", timeout=15)
        if r.status_code == 200:
            logs = r.json()
            if logs["summary"]:
                st.write("**Summary by class:**")
                st.table(logs["summary"])
            if logs["recent"]:
                st.write("**Most recent:**")
                st.table(logs["recent"])
        else:
            st.caption("Could not load logs from backend.")
    except requests.RequestException:
        st.caption("Could not reach backend for logs.")


if __name__ == "__main__":
    main()
