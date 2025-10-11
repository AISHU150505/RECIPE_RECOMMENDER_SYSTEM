import streamlit as st
import pandas as pd
import numpy as np
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import scipy.sparse as sps  # <-- NEW

# MEMORY BASED - USER BASED
class UserBasedCF:
    """
    Memory-based (neighborhood) collaborative filtering:
      - user-user cosine similarity on (optionally mean-centered) ratings
      - prediction: mu_u + sum_v sim(u,v) * (r_vi - mu_v) / sum |sim(u,v)|
      - shrinkage to temper small co-rating overlaps
    """
    def __init__(self, k_neighbors=50, center=True, shrinkage=10.0, min_common=1,
                 rating_min=1.0, rating_max=5.0, clip=True):
        self.k = k_neighbors
        self.center = center
        self.shrinkage = shrinkage
        self.min_common = min_common
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.clip = clip

        # learned artifacts
        self.user_index = {}
        self.index_user = []
        self.item_index = {}
        self.index_item = []
        self.R = None
        self.user_means = None
        self.user_item_sets = None
        self.item_user_lists = None
        self.Su = None

        # stats
        self._co_counts = None   # keep as SPARSE matrix

    def fit(self, reviews_df: pd.DataFrame):
        # 1) build indices
        users = reviews_df['user_id'].astype(str).unique()
        items = reviews_df['recipe_id'].astype(str).unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.index_user = list(users)
        self.item_index = {it: j for j, it in enumerate(items)}
        self.index_item = list(items)

        # 2) sparse user-item matrix
        rows = reviews_df['user_id'].map(self.user_index).to_numpy()
        cols = reviews_df['recipe_id'].map(self.item_index).to_numpy()
        data = reviews_df['rating'].astype(float).to_numpy()
        nU, nI = len(users), len(items)
        R = csr_matrix((data, (rows, cols)), shape=(nU, nI))

        # 3) mean-center (optional)
        if self.center:
            user_mean_map = reviews_df.groupby('user_id')['rating'].mean().astype(float)
            self.user_means = np.array([user_mean_map.get(u, np.nan) for u in self.index_user], dtype=float)
            R = R.tolil(copy=True)
            for i in range(nU):
                if R.rows[i]:
                    R.data[i] = [x - self.user_means[i] for x in R.data[i]]
            R = R.tocsr()
        else:
            self.user_means = np.zeros(nU, dtype=float)

        self.R = R

        # 4) helper structures
        self.user_item_sets = [set(R[i].indices.tolist()) for i in range(nU)]
        self.item_user_lists = [R[:, j].nonzero()[0].tolist() for j in range(nI)]

        # 5) (Optional) precompute dense user-user cosine (beware memory if nU is large)
        self.Su = cosine_similarity(R)  # if memory is an issue, set self.Su = None

        # 6) sparse co-rating counts
        B = (R != 0).astype(np.int8)
        self._co_counts = (B @ B.T).tocsr()

        return self

    def _neighbors_for_item(self, u_idx, j_idx):
        """Return candidate neighbor user indices who rated item j."""
        cand = self.item_user_lists[j_idx]
        if not cand:
            return []

        # similarities for user u to each candidate v
        sims = self.Su[u_idx, cand] if self.Su is not None else cosine_similarity(self.R[u_idx], self.R[cand]).ravel()
        sims = np.asarray(sims).ravel()

        # co-rated counts for shrinkage/min_common (densify slice)
        if self._co_counts is not None:
            cc = self._co_counts[u_idx, cand]  # 1 x len(cand) sparse slice
            commons = cc.toarray().ravel().astype(float) if sps.issparse(cc) else np.asarray(cc).ravel()
        else:
            commons = np.array([len(self.user_item_sets[u_idx].intersection(self.user_item_sets[v]))
                                for v in cand], dtype=float)

        # filter by min_common
        mask = commons >= float(self.min_common)
        if not mask.any():
            return []

        cand = np.asarray(cand)[mask]
        sims = sims[mask]
        commons = commons[mask]

        # shrinkage: sim' = sim * (n / (n + shrink))
        if self.shrinkage > 0:
            sims = sims * (commons / (commons + float(self.shrinkage)))

        # top-K by |sim|
        if sims.size > self.k:
            top_idx = np.argpartition(np.abs(sims), -self.k)[-self.k:]
            cand = cand[top_idx]
            sims = sims[top_idx]

        order = np.argsort(-np.abs(sims))
        return list(zip(cand[order], sims[order]))

    def _predict_one(self, u_idx, j_idx):
        """Predict rating for known user u on item j."""
        neigh = self._neighbors_for_item(u_idx, j_idx)
        if not neigh:
            base = self.user_means[u_idx] if not np.isnan(self.user_means[u_idx]) else 3.5
            return float(base)

        num = 0.0
        den = 0.0
        for v_idx, sim_uv in neigh:
            rvj = self.R[v_idx, j_idx]
            if sps.issparse(rvj):  # very defensive; usually scalar already
                rvj = rvj.toarray().item()
            if rvj == 0:
                continue
            num += sim_uv * rvj
            den += abs(sim_uv)

        base = self.user_means[u_idx] if self.center else 0.0
        pred = base if den == 0 else base + (num / den)
        return float(np.clip(pred, self.rating_min, self.rating_max) if self.clip else pred)

    def predict_many(self, user_id, recipe_ids):
        """Returns a dict {recipe_id: predicted_score}."""
        out = {}
        u_idx = self.user_index.get(str(user_id))
        if u_idx is None:
            prior = 3.5
            for rid in recipe_ids:
                out[rid] = float(prior)
            return out

        for rid in recipe_ids:
            j_idx = self.item_index.get(str(rid))
            if j_idx is None:
                base = self.user_means[u_idx] if not np.isnan(self.user_means[u_idx]) else 3.5
                out[rid] = float(base)
            else:
                out[rid] = self._predict_one(u_idx, j_idx)
        return out



from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# ---- Drop-in robust parquet loader + path guard ----
import os, io, re, ast, csv, duckdb
import numpy as np
import pandas as pd



def _has_parquet_magic(path: str) -> bool:
    """True if file exists and starts/ends with 'PAR1'."""
    if not (path and os.path.exists(path) and os.path.isfile(path)):
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(-4, io.SEEK_END)
            tail = f.read(4)
        return head == b"PAR1" and tail == b"PAR1"
    except Exception:
        return False

def load_parquet_robust(path: str) -> pd.DataFrame:
    """Try pyarrow → fastparquet → DuckDB. If file isn't valid Parquet, raise."""
    # Quick sanity: reject if file clearly isn't Parquet
    if not _has_parquet_magic(path):
        raise ValueError(f"File is not valid Parquet (no magic bytes): {path}")

    # 1) pyarrow
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as e1:
        # 2) fastparquet (optional)
        try:
              # noqa: F401
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as e2:
            # 3) DuckDB
            try:
                return duckdb.sql(f"SELECT * FROM read_parquet('{path}')").df()
            except Exception as e3:
                raise RuntimeError(
                    "All parquet readers failed.\n"
                    f"pyarrow: {type(e1).__name__}: {e1}\n"
                    f"fastparquet: {type(e2).__name__}: {e2}\n"
                    f"duckdb: {type(e3).__name__}: {e3}\n"
                    f"Path tried: {path}"
                )

# ----------------- your existing helpers (unchanged) -----------------
def parse_listish(s):
    if isinstance(s, list): return [str(x).strip() for x in s]
    if isinstance(s, (int, float)) and pd.isna(s): return []
    if not isinstance(s, str): return []
    t = s.strip()
    if not t: return []
    t = re.sub(r'^\s*c\(', '[', t); t = re.sub(r'\)\s*$', ']', t)
    if (t.startswith('[') and t.endswith(']')) or (t.startswith('(') and t.endswith(')')):
        try:
            val = ast.literal_eval(t)
            if isinstance(val, (list, tuple)):
                return [str(x).strip() for x in val]
        except Exception:
            pass
    return [x.strip() for x in re.split(r"[,\s;|/]+", t) if x.strip()]

def iso8601_to_minutes(x):
    if not isinstance(x, str): return np.nan
    m = re.match(r'P?T?(?:(\d+)H)?(?:(\d+)M)?$', x.strip(), re.I)
    if not m: return np.nan
    h = int(m.group(1)) if m.group(1) else 0
    mm = int(m.group(2)) if m.group(2) else 0
    return h*60 + mm


def _normalize_recipes(df: pd.DataFrame) -> pd.DataFrame:
    if 'id' in df.columns and 'minutes' in df.columns:
        out = pd.DataFrame({
            'recipe_id': df['id'].astype(str),
            'name': df['name'].astype(str),
            'minutes': pd.to_numeric(df['minutes'], errors='coerce'),
            'tags': (df['tags'] if 'tags' in df.columns else pd.Series([""]*len(df))).S(parse_listish),
            'ingredients': (df['ingredients'] if 'ingredients' in df.columns else pd.Series([""]*len(df))).apply(parse_listish),
            'description': (df['description'] if 'description' in df.columns else pd.Series([""]*len(df))).astype(str)
        })
        if 'nutrition' in df.columns:
            def _nut(v):
                try:
                    arr = list(ast.literal_eval(v)) if isinstance(v, str) else (v if isinstance(v, list) else [])
                except Exception:
                    arr = []
                arr = (arr + [np.nan]*7)[:7]
                return pd.Series(arr, index=['calories','fat','sugar','sodium','protein','sat_fat','carbs'])
            nut = df['nutrition'].apply(_nut)
            out = pd.concat([out, nut[['calories','fat','sat_fat','sodium','carbs','sugar','protein']]], axis=1)
        for c in ['calories','fat','sat_fat','cholesterol','sodium','carbs','fiber','sugar','protein']:
            if c not in out.columns: out[c] = np.nan
        return out

    rename = {
        'RecipeId':'recipe_id','Name':'name','Description':'description',
        'Keywords':'tags_raw','RecipeIngredientParts':'ingredients_raw',
        'TotalTime':'total_iso','CookTime':'cook_iso','PrepTime':'prep_iso',
        'Calories':'calories','FatContent':'fat','SaturatedFatContent':'sat_fat',
        'CholesterolContent':'cholesterol','SodiumContent':'sodium',
        'CarbohydrateContent':'carbs','FiberContent':'fiber',
        'SugarContent':'sugar','ProteinContent':'protein'
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    assert {'recipe_id','name'}.issubset(df.columns), f"Missing recipe_id/name in recipes. Found: {df.columns.tolist()}"

    total_m = (df['total_iso'] if 'total_iso' in df.columns else pd.Series([np.nan]*len(df))).apply(iso8601_to_minutes)
    cook_m  = (df['cook_iso']  if 'cook_iso'  in df.columns else pd.Series([0]*len(df))).apply(iso8601_to_minutes)
    prep_m  = (df['prep_iso']  if 'prep_iso'  in df.columns else pd.Series([0]*len(df))).apply(iso8601_to_minutes)
    minutes = total_m.where(~total_m.isna(), cook_m.add(prep_m, fill_value=0))
    minutes = minutes.fillna(minutes.median() if minutes.notna().any() else 30)

    tags = (df['tags_raw'] if 'tags_raw' in df.columns else pd.Series([""]*len(df))).apply(parse_listish)
    ings = (df['ingredients_raw'] if 'ingredients_raw' in df.columns else pd.Series([""]*len(df))).apply(parse_listish)
    desc = (df['description'] if 'description' in df.columns else pd.Series([""]*len(df))).astype(str)

    out = pd.DataFrame({
        'recipe_id': df['recipe_id'].astype(str),
        'name': df['name'].astype(str),
        'minutes': minutes.astype(float),
        'tags': tags,
        'ingredients': ings,
        'description': desc,
        'calories': pd.to_numeric(df['calories'] if 'calories' in df.columns else pd.Series([np.nan]*len(df)), errors='coerce'),
        'fat': pd.to_numeric(df['fat'] if 'fat' in df.columns else pd.Series([np.nan]*len(df)), errors='coerce'),
        'sat_fat': pd.to_numeric(df['sat_fat'] if 'sat_fat' in df.columns else pd.Series([np.nan]*len(df)), errors='coerce'),
        'cholesterol': pd.to_numeric(df['cholesterol'] if 'cholesterol' in df.columns else pd.Series([np.nan]*len(df)), errors='coerce'),
        'sodium': pd.to_numeric(df['sodium'] if 'sodium' in df.columns else pd.Series([np.nan]*len(df)), errors='coerce'),
        'carbs': pd.to_numeric(df['carbs'] if 'carbs' in df.columns else pd.Series([np.nan]*len(df)), errors='coerce'),
        'fiber': pd.to_numeric(df['fiber'] if 'fiber' in df.columns else pd.Series([np.nan]*len(df)), errors='coerce'),
        'sugar': pd.to_numeric(df['sugar'] if 'sugar' in df.columns else pd.Series([np.nan]*len(df)), errors='coerce'),
        'protein': pd.to_numeric(df['protein'] if 'protein' in df.columns else pd.Series([np.nan]*len(df)), errors='coerce'),
    })
    return out

def _normalize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    if {'user_id','recipe_id','rating'}.issubset(df.columns):
        out = df.copy()
    else:
        out = df.rename(columns={'AuthorId':'user_id','RecipeId':'recipe_id','Rating':'rating',
                                 'Review':'review_text','DateSubmitted':'date'})
        keep = [c for c in ['user_id','recipe_id','rating','review_text','date'] if c in out.columns]
        out = out[keep]
    out['user_id']   = out['user_id'].astype(str)
    out['recipe_id'] = out['recipe_id'].astype(str)
    out['rating']    = pd.to_numeric(out['rating'], errors='coerce')
    out = out[out['rating'].between(1,5)]
    return out
import pandas as pd
from sqlalchemy import create_engine

# --- MySQL Configuration ---
MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "food_recommneder_data"
TABLE_NAME = "recipes"

# --- Create SQLAlchemy engine ---
engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)
recipes_df = pd.read_sql(f"SELECT * FROM recipes", con=engine)
print("recipes table →", len(recipes_df.columns), "columns")
print("Column names:", recipes_df.columns.tolist())

# --- Check reviews.csv ---
# reviews_df = pd.read_csv(REVIEWS_CSV)  # read only first row
reviews_df = pd.read_sql(f"SELECT * FROM reviews", con=engine)
def load_recipes_any():
    return _normalize_recipes(recipes_df)

def load_reviews_any():
    return _normalize_reviews(reviews_df)

# ---- run loads ----
recipes = load_recipes_any()
reviews = load_reviews_any()

recipes_used = recipes[recipes['recipe_id'].isin(reviews['recipe_id'].unique())].copy()
#MATRIX FACTORIZATION - MODEL BASED
class CFModelNMF:
    def __init__(self, n_components=60, random_state=42, max_iter=500, init="nndsvd"):
        self.k=n_components; self.random_state=random_state; self.max_iter=max_iter; self.init=init
        self.user_index={}; self.item_index={}; self.U=None; self.V=None
    def fit(self, reviews_df):
        users = reviews_df['user_id'].astype(str).unique()
        items = reviews_df['recipe_id'].astype(str).unique()
        self.user_index = {u:i for i,u in enumerate(users)}
        self.item_index = {it:i for i,it in enumerate(items)}
        rows = reviews_df['user_id'].map(self.user_index).values
        cols = reviews_df['recipe_id'].map(self.item_index).values
        data = reviews_df['rating'].astype(float).values
        R = csr_matrix((data,(rows,cols)), shape=(len(users), len(items)))
        nmf = NMF(n_components=self.k, init=self.init, random_state=self.random_state, max_iter=self.max_iter)
        self.U = nmf.fit_transform(R); self.V = nmf.components_.T
        return self
    def predict_many(self, user_id, recipe_ids):
        if user_id not in self.user_index: return {rid:0.0 for rid in recipe_ids}
        u = self.U[self.user_index[user_id]]
        out={}
        for rid in recipe_ids:
            j=self.item_index.get(str(rid))
            out[rid]=float(np.dot(u,self.V[j])) if j is not None else 0.0
        return out
    
# ===========================
# ✅ Save Trained Artifacts
# ===========================
from sklearn.model_selection import train_test_split


# ---------- 7) Train/test split & CF filter (re-run because IDs changed) ----------
from sklearn.model_selection import train_test_split
MIN_USER_RATINGS = 5; RANDOM_STATE = 42; MIN_RATINGS_CF = 5
def stratified_user_split(rev, test_size=0.2, min_ratings=MIN_USER_RATINGS):
    tr, te = [], []
    for uid, g in rev.groupby('user_id'):
        if len(g) < min_ratings: continue
        a,b = train_test_split(g, test_size=test_size, random_state=RANDOM_STATE)
        tr.append(a); te.append(b)
    return (pd.concat(tr).reset_index(drop=True), pd.concat(te).reset_index(drop=True)) if tr else (pd.DataFrame(),pd.DataFrame())

train, test = stratified_user_split(reviews)
u_ct = train['user_id'].value_counts(); i_ct = train['recipe_id'].value_counts()
active_users = u_ct[u_ct>=MIN_RATINGS_CF].index; active_items = i_ct[i_ct>=MIN_RATINGS_CF].index
train_cf = train[train['user_id'].isin(active_users) & train['recipe_id'].isin(active_items)].reset_index(drop=True)

# ---------- Train User-Based CF ----------
cf1 = UserBasedCF(
    k_neighbors=50,   # top-K neighbors
    center=True,      # mean-center ratings per user
    shrinkage=20.0,   # temper small overlaps
    min_common=2,     # require at least 2 co-rated items
    clip=True         # clip predictions within rating scale
).fit(train_cf)
# =========================
# CF + KB Hybrid: Clean, Self-Contained, No Globals Missing
# =========================

import numpy as np
import pandas as pd

# ---- 1) Train User-Based CF on your train_cf (must exist) ----
cf_model = UserBasedCF(
    k_neighbors=50,
    center=True,
    shrinkage=20.0,
    min_common=2,
    clip=True
).fit(train_cf)

# ---- 2) Build metadata lookup (name/minutes/calories) ----
recipes_lookup = (recipes_used
                  .copy()
                  .assign(recipe_id=lambda d: d["recipe_id"].astype(str))
                  .set_index("recipe_id"))[["name","minutes","calories"]]

# ---- 3) Popularity baseline for cold-start (mean * log(1+count)) ----
pop = (reviews.copy()
       .assign(recipe_id=lambda d: d["recipe_id"].astype(str))
       .groupby("recipe_id")["rating"]
       .agg(["mean","count"])
       .rename(columns={"mean":"pop_mean","count":"pop_cnt"}))
pop["pop_score"] = pop["pop_mean"].fillna(0.0) * np.log1p(pop["pop_cnt"].fillna(0.0))



recipes_lookup = recipes.set_index("recipe_id")[["name","minutes","calories"]]
cf_model = UserBasedCF(
    k_neighbors=50,
    center=True,
    shrinkage=20.0,
    min_common=2,
    clip=True
).fit(train_cf)
# Save everything once
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, text_cols=None, max_features=10000):
        self.text_cols = text_cols or ["ingredients", "tags", "description"]
        self.max_features = max_features
        self.vectorizer = None
        self.tfidf_matrix = None
        self.recipe_ids = None

    # ------------------------------------------------------------
    def _safe_join(self, x):
        """Join safely whether x is list, numpy array, or string."""
        import ast, numbers
        # None or NaN
        if x is None:
            return ""
        if isinstance(x, numbers.Number):
            return str(x)
        # List or array
        if isinstance(x, (list, tuple, np.ndarray)):
            return " ".join(map(str, x))
        # String that looks like a list
        s = str(x).strip()
        if not s:
            return ""
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return " ".join(map(str, parsed))
            except Exception:
                pass
        return s

    # ------------------------------------------------------------
    def _combine_text(self, df):
        parts = []
        for col in self.text_cols:
            if col not in df.columns:
                continue
            col_text = df[col].apply(self._safe_join)
            parts.append(col_text)
        # join all text sources together
        return pd.Series([" ".join(x) for x in zip(*parts)], index=df.index)

    # ------------------------------------------------------------
    def fit(self, recipes_df):
        """Build TF-IDF embeddings for recipes."""
        print("Building TF-IDF embeddings for recipes...")
        self.recipe_ids = recipes_df["recipe_id"].astype(str).tolist()
        texts = self._combine_text(recipes_df)

        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        print(f"TF-IDF built for {self.tfidf_matrix.shape[0]} recipes × {self.tfidf_matrix.shape[1]} features")
        return self

    # ------------------------------------------------------------
    def recommend_similar(self, recipe_id, topn=10):
        """Recommend similar recipes given a recipe_id."""
        if self.tfidf_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if recipe_id not in self.recipe_ids:
            return pd.DataFrame(columns=["recipe_id", "similarity"])

        idx = self.recipe_ids.index(recipe_id)
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).ravel()
        top_idx = np.argsort(-sims)[1: topn + 1]
        return pd.DataFrame({
            "recipe_id": [self.recipe_ids[i] for i in top_idx],
            "similarity": sims[top_idx]
        })

    # ------------------------------------------------------------
    def recommend_for_user(self, user_id, reviews_df, recipes_df, topn=10):
        """
        Recommend recipes for a user based on high-rated items (>=4).
        Combines the TF-IDF vectors of liked recipes to compute new recommendations.
        """
        liked = reviews_df[
            (reviews_df["user_id"].astype(str) == str(user_id)) &
            (reviews_df["rating"] >= 3)
        ]["recipe_id"].astype(str).tolist()

        if not liked:
            print(f"⚠️ No liked recipes for user {user_id}")
            return pd.DataFrame(columns=["recipe_id", "score"])

        liked_idx = [self.recipe_ids.index(rid) for rid in liked if rid in self.recipe_ids]
        if not liked_idx:
            print(f"⚠️ None of user {user_id}'s liked recipes found in TF-IDF index.")
            return pd.DataFrame(columns=["recipe_id", "score"])

        # Build average vector for the user's preferences
        user_vec = np.mean(self.tfidf_matrix[liked_idx].toarray(), axis=0).reshape(1, -1)
        sims = cosine_similarity(user_vec, self.tfidf_matrix).ravel()

        recs = pd.DataFrame({"recipe_id": self.recipe_ids, "score": sims})
        recs = recs[~recs["recipe_id"].isin(liked)]  # exclude already liked
        top = recs.nlargest(topn, "score")

        # Join with metadata
        top = top.merge(recipes_df[["recipe_id", "name", "minutes", "calories"]],
                        on="recipe_id", how="left")

        return top.reset_index(drop=True)

# Initialize model
cb_model = ContentBasedRecommender(
    text_cols=["ingredients", "tags", "description", "doc_text"],  # choose text columns
    max_features=10000
)

# Train it
cb_model.fit(recipes)
def normalize_dict(d):
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    min_v, max_v = vals.min(), vals.max()
    if max_v > min_v:
        return {k: (v - min_v) / (max_v - min_v + 1e-8) for k, v in d.items()}
    return {k: 0.0 for k in d}
def fuse_scores_cf_cb_kb(
    user_id,
    candidate_ids,
    cf,
    cb_model,
    kb_bonus,
    reviews_df,
    recipes_df,
    alpha=0.6,  # CF weight
    beta=0.3,   # CB weight
    gamma=0.1,  # KB weight
    cold_min=5,
    user_hist_len=0
):
    """
    Hybrid fusion of User-Based CF + Content-Based + KB filtering.

    Parameters:
    ----------
    user_id : str
        Current user ID.
    candidate_ids : list[str]
        Recipe IDs to score.
    cf : UserBasedCF
        Trained collaborative filtering model.
    cb_model : ContentBasedRecommender
        Trained content-based recommender.
    kb_bonus : dict
        Knowledge-based score boosts {recipe_id: bonus}.
    reviews_df : pd.DataFrame
        User–recipe ratings dataframe (for user profile in CB).
    recipes_df : pd.DataFrame
        Recipes dataframe (for CB text lookup).
    alpha, beta, gamma : float
        Weights for CF, CB, and KB components respectively.
    cold_min : int
        If user has < cold_min ratings, reduce CF weight.
    user_hist_len : int
        Number of ratings user has in training data.
    """

    # 🧊 Cold-start handling
    if user_hist_len < cold_min:
        alpha, beta, gamma = 0.3, 0.5, 0.2

    # ---------------- CF PREDICTIONS ----------------
    cf_scores = cf.predict_many(user_id, candidate_ids)

    # ---------------- CB PREDICTIONS ----------------
    cb_scores_df = cb_model.recommend_for_user(
        user_id=user_id,
        reviews_df=reviews_df,
        recipes_df=recipes_df,
        topn=len(candidate_ids)
    )
    cb_scores = dict(zip(cb_scores_df["recipe_id"].astype(str), cb_scores_df["score"]))
    cf_scores = normalize_dict(cf_scores)
    cb_scores = normalize_dict(cb_scores)
    kb_bonus  = normalize_dict(kb_bonus)
    # ---------------- FUSION ----------------
    fused = {}
    for rid in candidate_ids:
        cf_score = cf_scores.get(rid, 0.0)
        cb_score = cb_scores.get(rid, 0.0)
        kb_score = kb_bonus.get(rid, 0.0)
        fused[rid] = alpha * cf_score + beta * cb_score + gamma * kb_score

    # Return as sorted dict by score
    return dict(sorted(fused.items(), key=lambda kv: kv[1], reverse=True))


with open("cf_cb_kb_hybrid1.pkl", "wb") as f:
    pickle.dump({
        "cf_model": cf_model,
        "recipes_lookup": recipes_lookup,
        "pop": pop,"cb_model": cb_model,  
        "train": train,
        "test": test
    }, f)

print("✅ Model saved successfully → cf_kb_hybrid.pkl")
