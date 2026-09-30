# Fake News Detection — NLP + SHAP

TF-IDF + Random Forest pipeline to classify news as real or fake, 
with SHAP explainability and a live Streamlit demo.

## Pipeline
Text cleaning → TF-IDF (unigrams+bigrams, 20k features) → Random Forest (200 trees)

## Dataset
WELFake (~63.5k deduplicated articles)

## Results
~91% test accuracy — see notebooks/01_training_data_completed.ipynb for full report

## Explainability
SHAP TreeExplainer highlights which words push a prediction toward fake vs real.

## Demo
[Live app](your-streamlit-cloud-link-here)

## Setup
\`\`\`
pip install -r requirements.txt
streamlit run app.py
\`\`\`