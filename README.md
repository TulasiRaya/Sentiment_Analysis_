# SentimentML — Streamlit Edition

Pure-Python rewrite of the original Flask + HTML/JS/CSS app. No JavaScript, no
templates, no static files — everything (UI, ML training, predictions) runs in
one `app.py` powered by Streamlit.

## What it does

- **Live text box**: paste any text and get an instant Positive/Negative/Neutral
  prediction with a sentence-level breakdown (heuristic/keyword based, same
  logic as the original `/predict_text` route).
- **File upload + training**: upload a CSV, XLSX, JSON, PDF, DOCX, or TXT file.
  - Structured files (CSV/XLSX/JSON) with a rating/sentiment column train a
    real **TF-IDF + Logistic Regression** classifier on your labels.
  - Unstructured files (PDF/DOCX/TXT) or structured files without a label
    column get pseudo-labels from a keyword heuristic, then train the same
    ML pipeline on top of that.
- **Results dashboard**: colored metric cards (accuracy, positive/negative/
  neutral counts, total reviewed), a sentiment-distribution donut chart, a
  per-class F1 bar chart, and a filterable, paginated predictions table
  (with match ✓/✗ and confidence bars) you can download as CSV.
- **Light / Dark mode**: a toggle in the top-right switches the entire
  dashboard's color scheme (cards, badges, charts, table) between a light and
  a dark theme.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Notes on the rewrite

- All the original Flask route logic (`clean_text`, `normalize_label`,
  `heuristic_sentiment`, the file parsers, and the `/predict` training
  pipeline) was ported as-is into plain Python functions — same behavior,
  no JS required.
- Charts use Plotly instead of Chart.js.
- State (the trained results) is kept in `st.session_state` so the dashboard
  persists across reruns until you upload/train a new file.
