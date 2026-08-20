# Surface Crack Defect Classifier

A CNN that classifies concrete surface images as `crack` / `no_crack`,
served as a REST API and containerized with Docker — the "build → train →
deploy" pipeline using Python, SQL, and TensorFlow, with a production-style
deployment pattern (FastAPI + Docker) rather than just a notebook demo.

## Problem

Manual visual inspection for structural defects (cracks in concrete,
pavement, etc.) is slow and inconsistent. An automated classifier can screen
large batches of images and flag candidates for human review.

## Architecture

```
Image upload → FastAPI /predict → CNN (TensorFlow) → prediction + confidence
                                          │
                                    SQLite (SQL logging of every request)
                                          │
                                   GET /logs (SQL summary of history)
```

- **Data**: Synthetic concrete-texture images with randomly drawn crack
  lines, standing in for the real
  [Kaggle Surface Crack Detection dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)
  (40,000 labeled images). See `src/generate_data.py` for how to swap in the
  real dataset — same folder structure, no downstream changes needed.
- **Model**: Small CNN (3 conv blocks + dense head) trained from scratch —
  98.3% validation accuracy on the synthetic set. `src/model.py` /
  `src/train.py` also documents how to switch to MobileNetV2 transfer
  learning for better generalization to real photos, once you have
  unrestricted internet access to download ImageNet weights.
- **SQL**: Every prediction (filename, predicted class, confidence,
  timestamp) is logged to SQLite via `api/db.py`. `GET /logs` returns both
  the most recent predictions and a SQL `GROUP BY` summary per class —
  this is the SQL skill-building part of the project, not just storage.
- **Deployment**: FastAPI service (`api/main.py`) + `Dockerfile`, so it runs
  identically locally or on any container platform.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/predict` | POST | Upload an image (`file` form field) → `{predicted_class, confidence, raw_score}` |
| `/logs` | GET | Recent predictions + per-class SQL summary |

Interactive API docs (Swagger UI) are auto-generated at `/docs` once running.

## Project structure

```
crack-defect-classifier/
├── src/
│   ├── generate_data.py   # synthetic image generator (swap for real Kaggle data)
│   ├── model.py            # CNN architecture (+ transfer-learning notes)
│   └── train.py             # training + evaluation
├── api/
│   ├── main.py              # FastAPI app
│   └── db.py                 # SQLite logging + SQL summary queries
├── models/                  # trained model + config (committed, ~360KB)
├── data/                    # images + SQLite log (gitignored, regenerate locally)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Running locally (no Docker)

```bash
git clone <this-repo-url>
cd crack-defect-classifier
pip install -r requirements.txt

# Regenerate data + retrain (optional — a trained model is already committed)
python src/generate_data.py
python src/train.py

# Launch the API
uvicorn api.main:app --reload --port 8000
```

Test it:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@data/images/crack/crack_0000.png"
```

Or open `http://localhost:8000/docs` for the interactive Swagger UI —
upload an image directly in the browser.

## Running with Docker

```bash
docker build -t crack-classifier .
docker run -p 8000:8000 crack-classifier
```

Then hit the same endpoints at `http://localhost:8000`.

## Deployment

Deployed on [Render](https://render.com)'s free web-service tier (supports
Docker deploys from a GitHub repo directly — no credit card required for the
free plan, though the free tier spins down after inactivity and takes ~30s
to wake up on the next request).

To deploy your own copy: create a new **Web Service** on Render, connect
this GitHub repo, choose **Docker** as the environment, and it builds
automatically from the `Dockerfile`.

## Notes on the synthetic data

`kaggle.com` wasn't reachable from the environment this was built in, so
`src/generate_data.py` creates textured backgrounds with randomly drawn
crack lines instead of using the real dataset. The model's 98.3% accuracy
reflects performance on this synthetic task, not validated performance on
real crack photos — swapping in the real Kaggle dataset (same folder
structure) is a drop-in change with no code changes required downstream.
