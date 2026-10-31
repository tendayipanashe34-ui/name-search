# Cooperative Name Search: web app

Streamlit app for AI-assisted cooperative / member name search and duplicate detection.
Built from `Cooperative_Name_Search_Notebook.ipynb` (same features, models and bands).

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. A trained model is included, so it works immediately.

## Pages
| Page | Purpose |
|---|---|
| Search | Ranked matches for a (misspelled) name, with probability, confidence band and evidence; send results to the review queue |
| Check a proposed name | Screen a new cooperative name before registration |
| Review queue | Officer marks each pair Same / Different / Needs more information; save or download |
| Dashboard | Register size, completeness, duplicate risk by region and type (filters) |
| Model performance | Metrics, retrieval vs exact-match baseline, feature importance, ROC/PR curves |
| Data & retrain | Upload the approved anonymised register (`name`, optional `region`) and retrain |

## Deploy (Streamlit Community Cloud)
Push this folder to GitHub, create a new app pointing at `app.py`. `requirements.txt` pins
scikit-learn to the version the bundled model was trained with. If you change versions, use
**Data & retrain** to rebuild the model.

## Files
- `app.py`: the web app
- `coop_name_search.py`: engine (normalisation, features, models, search); also a CLI
- `coop_search_outputs/`: trained model, metrics, synthetic register, sample review queue
- `data/`: Ushirika regional statistics used to shape the synthetic register

## Responsible use
Similarity is evidence, not proof of identity. No automatic merge, delete or approval; an officer
records every decision. The bundled register is **synthetic** (the Ushirika CSVs contain no names) and
reported scores come from simulated duplicates. Use only an approved, anonymised extract and validate on
officer-labelled real pairs before relying on accuracy figures.
