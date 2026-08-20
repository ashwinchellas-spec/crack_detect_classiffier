"""
SQL logging for the crack-classifier API. Every inference request gets
logged to SQLite: filename, predicted class, confidence, and timestamp.
This is the SQL piece of this project -- it shows the model wired into a
real data pipeline instead of just returning a prediction and forgetting it.
"""

import sqlite3
import os
from datetime import datetime, timezone

DB_PATH = "data/inference_log.db"


def init_db(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(filename, predicted_class, confidence, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO predictions (filename, predicted_class, confidence, timestamp) "
        "VALUES (?, ?, ?, ?)",
        (filename, predicted_class, confidence, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit=20, db_path=DB_PATH):
    """SQL-based summary: most recent predictions, and a per-class breakdown."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        "SELECT filename, predicted_class, confidence, timestamp "
        "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    )
    recent = cur.fetchall()

    cur.execute(
        "SELECT predicted_class, COUNT(*) as n, AVG(confidence) as avg_conf "
        "FROM predictions GROUP BY predicted_class"
    )
    summary = cur.fetchall()

    conn.close()
    return {
        "recent": [
            {"filename": r[0], "predicted_class": r[1], "confidence": r[2], "timestamp": r[3]}
            for r in recent
        ],
        "summary": [
            {"predicted_class": s[0], "count": s[1], "avg_confidence": s[2]}
            for s in summary
        ],
    }
