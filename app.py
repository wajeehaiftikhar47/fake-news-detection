"""
Fake News Detector — Streamlit app
Loads the saved TF-IDF + Random Forest pipeline and explains each
prediction with SHAP.

Expected files (relative to this app, e.g. in a sibling ../model/ folder,
adjust MODEL_DIR below to match your repo layout):
    model/vectorizer.pkl
    model/rf_model.pkl
    model/shap_background.pkl
"""

import re
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
import matplotlib.pyplot as plt

MODEL_DIR = "model"  # change to "../model" if running app.py from a subfolder


@st.cache_resource
def load_artifacts():
    vec = joblib.load(f"{MODEL_DIR}/vectorizer.pkl")
    rf = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
    bg_bundle = joblib.load(f"{MODEL_DIR}/shap_background.pkl")
    explainer = shap.TreeExplainer(
        rf, bg_bundle["background"], feature_names=bg_bundle["feature_names"]
    )
    return vec, rf, explainer


def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\b(reuters|new york times|nyt|breitbart|video)\b", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.caption("TF-IDF + Random Forest, explained with SHAP · trained on the WELFake dataset")

vec, rf, explainer = load_artifacts()

text_input = st.text_area(
    "Paste a headline + article text",
    height=200,
    placeholder="Type or paste a news article here...",
)

top_n = st.slider("How many top words to explain", 5, 20, 10)

if st.button("Analyze", type="primary") and text_input.strip():
    cleaned = clean_text(text_input)

    if len(cleaned) < 20:
        st.warning("Text too short after cleaning — paste a fuller article for a reliable prediction.")
    else:
        X = vec.transform([cleaned])
        proba = rf.predict_proba(X)[0]
        pred = rf.predict(X)[0]

        label = "🔴 Likely FAKE" if pred == 1 else "🟢 Likely REAL"
        confidence = proba[pred]

        st.subheader(label)
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(float(confidence))

        with st.spinner("Computing SHAP explanation..."):
            sv = explainer(X.toarray(), check_additivity=False)
            sv_fake = sv[..., 1] if sv.values.ndim == 3 else sv

        st.subheader("Why the model made this call")
        fig, ax = plt.subplots(figsize=(7, 4))
        shap.plots.waterfall(sv_fake[0], max_display=top_n, show=False)
        st.pyplot(fig, bbox_inches="tight")
        plt.close(fig)

        st.caption(
            "Positive SHAP values push the prediction toward **Fake**; "
            "negative values push it toward **Real**."
        )

st.divider()
with st.expander("About this model"):
    st.markdown(
        """
- **Pipeline:** text cleaning → TF-IDF (unigrams+bigrams, 20,000 features) → Random Forest (200 trees)
- **Dataset:** WELFake (~63.5k deduplicated articles after cleaning)
- **Held-out test accuracy:** ~91% (see training notebook for the full classification report)
- **Explainability:** SHAP TreeExplainer, computed against a fixed 100-row background sample
        """
    )