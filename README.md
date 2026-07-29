# SentimentML — Streamlit Edition

A self-contained sentiment analysis dashboard built with Streamlit.
It lets you test text instantly and upload documents or datasets to train a
TF-IDF + Logistic Regression sentiment model, with an interactive dashboard
for results, charts, and downloadable predictions.

## Key features

- **Instant sentiment analysis** for any pasted text.
- **Multi-format file upload**: CSV, XLSX, JSON, PDF, DOCX, TXT.
- **Automatic model training** when labeled data is available.
- **Heuristic labeling fallback** for unstructured text files.
- **Visual dashboard** with accuracy, label counts, sentiment distribution,
  F1 score chart, and paginated prediction table.
- **Light / dark theme toggle** for easier viewing.

## Supported inputs

- CSV / XLSX / JSON: use a dataset with a text column and an optional label/
  rating column.
- TXT / PDF / DOCX: the app extracts text and applies a heuristic sentiment
  labeling model for training and prediction.

## How it works

1. Upload a supported file or paste text in the live analysis box.
2. If your file includes a rating/sentiment column, the app normalizes labels
   and trains a TF-IDF + Logistic Regression model.
3. If no label column is found, the app applies a keyword-based heuristic to
   generate sentiment labels and then trains the same ML pipeline.
4. The dashboard displays model accuracy, sentiment counts, charts, and a
   downloadable prediction table.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown by Streamlit (typically `http://localhost:8501`).

## Project structure

- `app.py` — main Streamlit application, data parsing, training, prediction,
  and UI.
- `requirements.txt` — Python package dependencies.

## Requirements

- Python 3.8+
- Streamlit
- pandas
- numpy
- scikit-learn
- plotly
- nltk
- PyPDF2
- python-docx

## Notes

- The app uses `st.session_state` to preserve results between reruns.
- For best model performance, use a structured dataset with a clear text column
  and a sentiment/rating label column.
- Unstructured documents rely on keyword heuristics when explicit labels are
  unavailable.
