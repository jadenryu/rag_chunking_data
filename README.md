# rag_chunking_data

Research code and data for evaluating RAG chunking strategies.

## Layout

```
src/        Python source (pipeline, evaluation, analysis, visualizations)
data/       processed_dataset.json and source datasets
results/    evaluation outputs, ANOVA/Tukey tables, summary stats
figures/    generated figures (PNG/SVG)
docs/       appendix and write-ups
```

## Running

Scripts use repo-root-relative paths, so run them **from the repo root**:

```bash
python src/main.py
python src/run_ragas_evaluation.py
python src/visualizations.py
```

Setup:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
