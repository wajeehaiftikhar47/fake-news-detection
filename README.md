# Fake News Detection — NLP + SHAP

TF-IDF + Random Forest pipeline to classify news as real or fake, 
with SHAP explainability and a live Streamlit demo.

## Pipeline
Text cleaning → TF-IDF (unigrams+bigrams, 20k features) → Random Forest (200 trees)

## Dataset
WELFake (~63.5k deduplicated articles)

## Results
89.08% test accuracy — see notebooks/01_training_data_completed.ipynb for full report

## Explainability
SHAP TreeExplainer highlights which words push a prediction toward fake vs real.

## Demo
[Live Demo](https://fake-news-detection-9rfdu3z5mf2umtvv3xeu8z.streamlit.app).

## Setup
\`\`\`
pip install -r requirements.txt
streamlit run app.py
\`\`\`
