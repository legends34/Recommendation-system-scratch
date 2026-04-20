"""
CineMatch - Flask Backend
Run: python app.py
Then open index.html in your browser.

Folder layout expected:
  my_recommender/
  ├── app.py
  ├── index.html
  ├── model_params.npz   <- export from Kaggle (see EXPORT CELL below)
  ├── model_misc.pkl     <- export from Kaggle (see EXPORT CELL below)
  ├── movies.csv
  └── ratings.csv

===========================================================================
EXPORT CELL — paste this into your Kaggle notebook and run it ONCE.
Then download model_params.npz and model_misc.pkl from Kaggle output.
===========================================================================

import pickle, numpy as np

np.savez('model_params.npz',
    X_m1=X_m1, X_u1=X_u1, b_m1=b_m1, b_u1=b_u1,
    X_g=X_g,   X_m3=X_m3, X_u3=X_u3, b_m3=b_m3, b_u3=b_u3,
    X_m4=X_m4, X_u4=X_u4, b_m4=b_m4, b_u4=b_u4,
    X_g5=X_g5, X_m5=X_m5, X_u5=X_u5, b_m5=b_m5, b_u5=b_u5,
    sim_matrix=sim_matrix,
    train_sim_preds_norm=train_sim_preds_norm,
    val_sim_preds_norm=val_sim_preds_norm,
    beta=np.array([beta_1, beta_2, beta_3, beta_4, beta_5]),
    scalars=np.array([mu, r_min, r_max])
)

with open('model_misc.pkl', 'wb') as f:
    pickle.dump({
        'me_classes': me.classes_,
        'ue_classes': ue.classes_,
        'ge_classes': ge.classes_,
        'movie_to_genres': {k: list(v) for k, v in movie_to_genres.items()},
        'user_movie_ratings': {u: dict(m) for u, m in user_movie_ratings.items()},
        'movie_raters': {m: list(r) for m, r in movie_raters.items()},
    }, f)

print("Done! Download model_params.npz and model_misc.pkl")

===========================================================================
"""

import os, pickle, time
import numpy as np
import pandas as pd
from collections import defaultdict
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask import Flask, jsonify, request, send_file
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# CONFIG — change paths here if your files are elsewhere
# ---------------------------------------------------------------------------
DATA_DIR       = "."          # folder containing movies.csv, ratings.csv
PARAMS_FILE    = os.path.join(DATA_DIR, "model_params.npz")
MISC_FILE      = os.path.join(DATA_DIR, "model_misc.pkl")
MOVIES_CSV     = os.path.join(DATA_DIR, "movies.csv")
RATINGS_CSV    = os.path.join(DATA_DIR, "ratings.csv")
# ---------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)   # allow index.html (any origin) to call the API

print("\n🎬  CineMatch — loading model (this takes ~30s on first run)...")
t0 = time.time()

# ── 1. Load CSVs ────────────────────────────────────────────────────────────
movies_df_raw = pd.read_csv(MOVIES_CSV)
ratings_raw   = pd.read_csv(RATINGS_CSV)

df = ratings_raw[["userId", "movieId", "rating"]].head(200_000)

# Cold-start filter — two passes to match training exactly
for _ in range(2):
    uc = df["userId"].value_counts()
    mc = df["movieId"].value_counts()
    df = df[df["userId"].isin(uc[uc >= 20].index)]
    df = df[df["movieId"].isin(mc[mc >= 10].index)]

print(f"   Dataset: {len(df):,} rows after cleaning")

# Train/val split — same seed as training
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_movies_set = set(train_df["movieId"])
train_users_set  = set(train_df["userId"])
val_df = val_df[
    val_df["movieId"].isin(train_movies_set) &
    val_df["userId"].isin(train_users_set)
]

# ── 2. Encoders ──────────────────────────────────────────────────────────────
me = LabelEncoder()
ue = LabelEncoder()
movie_train = me.fit_transform(train_df["movieId"])
user_train  = ue.fit_transform(train_df["userId"])
rating_train = train_df["rating"].values.astype(np.float32)

movie_val = me.transform(val_df["movieId"])
user_val  = ue.transform(val_df["userId"])
rating_val = val_df["rating"].values.astype(np.float32)

r_min = rating_train.min()
r_max = rating_train.max()
rating_train_norm = (rating_train - r_min) / (r_max - r_min)
mu = rating_train_norm.mean()

n_movies = len(me.classes_)
n_users  = len(ue.classes_)
k1 = k2 = k3 = k4 = k5 = 20

# ── 3. Load trained parameters ───────────────────────────────────────────────
print("   Loading trained model parameters...")
p = np.load(PARAMS_FILE)
X_m1, X_u1, b_m1, b_u1 = p["X_m1"], p["X_u1"], p["b_m1"], p["b_u1"]
X_g,  X_m3, X_u3, b_m3, b_u3 = p["X_g"], p["X_m3"], p["X_u3"], p["b_m3"], p["b_u3"]
X_m4, X_u4, b_m4, b_u4 = p["X_m4"], p["X_u4"], p["b_m4"], p["b_u4"]
X_g5, X_m5, X_u5, b_m5, b_u5 = p["X_g5"], p["X_m5"], p["X_u5"], p["b_m5"], p["b_u5"]
sim_matrix           = p["sim_matrix"]
train_sim_preds_norm = p["train_sim_preds_norm"]
val_sim_preds_norm   = p["val_sim_preds_norm"]
beta_1, beta_2, beta_3, beta_4, beta_5 = p["beta"]
mu_saved, r_min_saved, r_max_saved = p["scalars"]
# Use saved scalars so inference matches training exactly
mu, r_min, r_max = float(mu_saved), float(r_min_saved), float(r_max_saved)

with open(MISC_FILE, "rb") as f:
    misc = pickle.load(f)

movie_to_genres    = defaultdict(list, {int(k): list(v) for k, v in misc["movie_to_genres"].items()})
user_movie_ratings = defaultdict(dict, {int(u): dict(m) for u, m in misc["user_movie_ratings"].items()})
movie_raters       = defaultdict(list, {int(m): list(r) for m, r in misc["movie_raters"].items()})

# ── 4. Genre encoder ─────────────────────────────────────────────────────────
ge = LabelEncoder()
ge.classes_ = misc["ge_classes"]

# ── 5. Similarity cache ──────────────────────────────────────────────────────
print("   Building similarity cache...")
sim_raters_cache = {}

def precalculate_top20(user, movie):
    if (user, movie) in sim_raters_cache:
        return
    raters = movie_raters.get(movie, [])
    scored = []
    for r_user, _ in raters:
        if r_user == user:
            continue
        sim = sim_matrix[user][r_user]
        if sim > 0:
            scored.append((sim, r_user))
    scored = sorted(scored, key=lambda x: x[0], reverse=True)[:20]
    tot_sim = sum(s for s, _ in scored)
    sim_raters_cache[(user, movie)] = (scored, tot_sim)

for idx in range(len(rating_train)):
    precalculate_top20(user_train[idx], movie_train[idx])
for idx in range(len(rating_val)):
    precalculate_top20(user_val[idx], movie_val[idx])

print(f"   Ready in {time.time()-t0:.1f}s")

# ── 6. Prediction logic (unchanged from your notebook) ───────────────────────
def get_sim_pred(user, movie):
    raters = movie_raters[movie]
    if len(raters) == 0:
        return rating_train.mean()
    scored = []
    for r_user, rating in raters:
        if r_user == user:
            continue
        sim = sim_matrix[user][r_user]
        if sim > 0:
            scored.append((sim, rating))
    scored = sorted(scored, key=lambda x: x[0], reverse=True)[:20]
    tot_rating = tot_sim = 0.0
    for sim, rating in scored:
        tot_rating += sim * rating
        tot_sim    += sim
    return tot_rating / tot_sim if tot_sim > 0 else rating_train.mean()

def get_sim_vector(user, movie, X_u_matrix, k):
    raters = movie_raters[movie]
    if len(raters) == 0:
        return np.zeros(k, dtype=np.float32)
    scored = []
    for r_user, rating in raters:
        if r_user == user:
            continue
        sim = sim_matrix[user][r_user]
        if sim > 0:
            scored.append((sim, r_user))
    scored = sorted(scored, key=lambda x: x[0], reverse=True)[:20]
    sim_vec = np.zeros(k, dtype=np.float32)
    tot = 0.0
    for sim, r_user in scored:
        if r_user < len(X_u_matrix):   # safety guard
            sim_vec += sim * X_u_matrix[r_user]
            tot += sim
    return sim_vec / tot if tot > 0 else np.zeros(k, dtype=np.float32)

def prediction(movie_idx, user_idx, model_type, val=False, fallback_idx=None):
    i, j = movie_idx, user_idx

    if model_type == 1:
        return np.dot(X_m1[i], X_u1[j]) + b_m1[i] + b_u1[j] + mu

    elif model_type == 2:
        if not val and fallback_idx is not None and fallback_idx < len(train_sim_preds_norm):
            return train_sim_preds_norm[fallback_idx]
        elif val and fallback_idx is not None and fallback_idx < len(val_sim_preds_norm):
            return val_sim_preds_norm[fallback_idx]
        else:
            raw_sim = get_sim_pred(j, i)
            return (raw_sim - r_min) / (r_max - r_min)

    elif model_type == 3:
        g_indices = movie_to_genres[i]
        genre_sum = np.sum(X_g[g_indices], axis=0) / max(len(g_indices), 1) if len(g_indices) > 0 else np.zeros(k3, dtype=np.float32)
        m_vec = X_m3[i] + genre_sum
        return np.dot(m_vec, X_u3[j]) + b_m3[i] + b_u3[j] + mu

    elif model_type == 4:
        scored, tot = sim_raters_cache.get((j, i), ([], 0.0))
        sim_vec = sum(sim * X_u4[r_user] for sim, r_user in scored if r_user < len(X_u4)) / tot if tot > 0 else np.zeros(k4, dtype=np.float32)
        u_vec = X_u4[j] + sim_vec
        return np.dot(X_m4[i], u_vec) + b_m4[i] + b_u4[j] + mu

    elif model_type == 5:
        g_indices = movie_to_genres[i]
        genre_sum = np.sum(X_g5[g_indices], axis=0) / max(len(g_indices), 1) if len(g_indices) > 0 else np.zeros(k5, dtype=np.float32)
        m_vec = X_m5[i] + genre_sum
        scored, tot = sim_raters_cache.get((j, i), ([], 0.0))
        sim_vec = sum(sim * X_u5[r_user] for sim, r_user in scored if r_user < len(X_u5)) / tot if tot > 0 else np.zeros(k5, dtype=np.float32)
        u_vec = X_u5[j] + sim_vec
        return np.dot(m_vec, u_vec) + b_m5[i] + b_u5[j] + mu

    elif model_type == 6:
        pred1 = np.dot(X_m1[i], X_u1[j]) + b_m1[i] + b_u1[j] + mu

        if not val and fallback_idx is not None and fallback_idx < len(train_sim_preds_norm):
            pred2 = train_sim_preds_norm[fallback_idx]
        elif val and fallback_idx is not None and fallback_idx < len(val_sim_preds_norm):
            pred2 = val_sim_preds_norm[fallback_idx]
        else:
            pred2 = (get_sim_pred(j, i) - r_min) / (r_max - r_min)

        g = movie_to_genres[i]
        gs3 = np.sum(X_g[g], axis=0) / max(len(g), 1) if len(g) > 0 else np.zeros(k3, dtype=np.float32)
        pred3 = np.dot(X_m3[i] + gs3, X_u3[j]) + b_m3[i] + b_u3[j] + mu

        sv4 = get_sim_vector(j, i, X_u4, k4)
        pred4 = np.dot(X_m4[i], X_u4[j] + sv4) + b_m4[i] + b_u4[j] + mu

        gs5 = np.sum(X_g5[g], axis=0) / max(len(g), 1) if len(g) > 0 else np.zeros(k5, dtype=np.float32)
        sv5 = get_sim_vector(j, i, X_u5, k5)
        pred5 = np.dot(X_m5[i] + gs5, X_u5[j] + sv5) + b_m5[i] + b_u5[j] + mu

        S = beta_1 + beta_2 + beta_3 + beta_4 + beta_5
        return (beta_1*pred1 + beta_2*pred2 + beta_3*pred3 + beta_4*pred4 + beta_5*pred5) / S

def denorm(pred):
    return float(np.clip(pred * (r_max - r_min) + r_min, r_min, r_max))

# ── 7. Helper: get user's watched movies ────────────────────────────────────
def get_watched(original_user_id, limit=50):
    rows = df[df["userId"] == original_user_id][["movieId", "rating"]] \
             .sort_values("rating", ascending=False) \
             .head(limit)
    result = []
    for _, row in rows.iterrows():
        mid = int(row["movieId"])
        info = movies_df_raw[movies_df_raw["movieId"] == mid]
        if len(info) == 0:
            continue
        result.append({
            "title":  info["title"].values[0],
            "genres": info["genres"].values[0].replace("|", ", "),
            "rating": float(row["rating"]),
        })
    return result

# ── 8. Helper: recommend ────────────────────────────────────────────────────
def recommend(original_user_id, n=20):
    u_enc = ue.transform([original_user_id])[0]
    seen  = set(df[df["userId"] == original_user_id]["movieId"])
    unseen_real = list(set(me.classes_) - seen)
    m_enc_list  = me.transform(unseen_real)

    preds = []
    for m_enc in m_enc_list:
        pred_norm = prediction(int(m_enc), int(u_enc), model_type=6,
                               val=False, fallback_idx=None)
        score = denorm(pred_norm)
        preds.append((int(m_enc), score))

    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:n]

    result = []
    for m_enc, score in top:
        real_id = int(me.inverse_transform([m_enc])[0])
        info    = movies_df_raw[movies_df_raw["movieId"] == real_id]
        if len(info) == 0:
            continue
        result.append({
            "title":  info["title"].values[0],
            "genres": info["genres"].values[0].replace("|", ", "),
            "score":  round(score, 2),
        })
    return result

# ── 9. Flask routes ──────────────────────────────────────────────────────────
@app.route("/api/validate/<int:user_id>")
def validate_user(user_id):
    """Check if user_id exists in the dataset."""
    exists = int(user_id) in set(df["userId"])
    if not exists:
        valid_sample = sorted(df["userId"].unique()[:20].tolist())
        return jsonify({"valid": False, "sample_ids": valid_sample}), 404
    return jsonify({"valid": True})

@app.route("/api/user/<int:user_id>/watched")
def watched(user_id):
    try:
        data = get_watched(user_id)
        return jsonify({"userId": user_id, "watched": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/user/<int:user_id>/recommend")
def recommendations(user_id):
    n = int(request.args.get("n", 10))
    try:
        data = recommend(user_id, n=n)
        return jsonify({"userId": user_id, "recommendations": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/users/sample")
def sample_users():
    """Return a few valid userIds to help with testing."""
    sample = sorted(df["userId"].unique()[:30].tolist())
    return jsonify({"sample": sample})
@app.route("/")
def serve_frontend():
    # This tells Flask to serve your index.html when you go to localhost:5001
    return send_file("index.html")

if __name__ == "__main__":
    print("\n✅  Server running at http://localhost:5001")
    print("   Open index.html in your browser.\n")
    app.run(debug=False, port=5001)
