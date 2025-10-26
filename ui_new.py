import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import scipy.sparse as sps  
from train_new import UserBasedCF, ContentBasedRecommender

from train_new import recipes_used
@st.cache_resource
def load_model():
    with open("cf_cb_kb_hybrid1.pkl", "rb") as f:
        data = pickle.load(f)
    return (
        data["cf_model"],
        data["cb_model"],        
        data["recipes_lookup"],
        data["pop"],
        data["train"],
        data["test"]
    )

cf_model, cb_model, recipes_lookup, pop, train, test = load_model()



def apply_kb(recipes_df, prefs):
    df = recipes_df.copy()
    for col in ["minutes", "calories", "protein", "fat"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    if "minutes" in df.columns:
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    if "calories" in df.columns:
        df["calories"] = pd.to_numeric(df["calories"], errors="coerce")

    if prefs.get("max_minutes") is not None:
        df = df[df["minutes"] <= float(prefs["max_minutes"])]
    if prefs.get("max_calories") is not None:
        df = df[df["calories"] <= float(prefs["max_calories"])]
    if prefs.get("min_protein") is not None and "protein" in df.columns:
        df = df[df["protein"] >= float(prefs["min_protein"])]

    if prefs.get("max_fat") is not None and "fat" in df.columns:
        df = df[df["fat"] <= float(prefs["max_fat"])]

    bonus = {}
    if prefs.get("cuisine"):
        txt = df["doc_text"].astype(str) if "doc_text" in df.columns else df["name"].astype(str)

        mask = txt.str.contains(str(prefs["cuisine"]), case=False, na=False)
        bonus.update({str(rid): 0.1 for rid in df.loc[mask, "recipe_id"].astype(str)})
    return df, bonus

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_cf_cb_kb(
    cf_model,
    cb_model,
    recipes_df,
    reviews_df,
    user_id: str | None = None,
    seed_recipe_ids: list[str] | None = None,
    prefs: dict | None = None,
    topn: int = 10,
    alpha_cf: float = 0.6,   # CF weight
    beta_cb: float = 0.3,    # CB weight
    gamma_kb: float = 0.1,   # KB weight
):
    prefs = prefs or {}
    filtered, kb_bonus = apply_kb(recipes_df, prefs)
    candidates = [str(r) for r in filtered["recipe_id"].astype(str).tolist()]
    if not candidates:
        return pd.DataFrame(columns=["recipe_id", "name", "minutes", "calories", "score"])
    seen = set()
    if user_id is not None:
        u = str(user_id)
        seen |= set(reviews_df.loc[reviews_df["user_id"].astype(str) == u, "recipe_id"].astype(str))
    if seed_recipe_ids:
        seen |= set(map(str, seed_recipe_ids))
    candidates = [c for c in candidates if c not in seen]
    if not candidates:
        return pd.DataFrame(columns=["recipe_id", "name", "minutes", "calories", "score"])
    hist_len = int((reviews_df["user_id"].astype(str) == str(user_id)).sum()) if user_id else 0
    if hist_len == 0:
        tmp = pd.DataFrame({"recipe_id": candidates})
        tmp = tmp.join(pop[["pop_score"]], on="recipe_id")
        tmp["pop_score"] = tmp["pop_score"].fillna(0.0)
        tmp["kb_bonus"] = tmp["recipe_id"].map(lambda x: kb_bonus.get(str(x), 0.0))
        tmp["score"] = 0.9 * tmp["pop_score"] + 0.1 * tmp["kb_bonus"]
        min_s, max_s = tmp["score"].min(), tmp["score"].max()
        if max_s > min_s:
            tmp["score"] = (tmp["score"] - min_s) / (max_s - min_s + 1e-8)
        else:
            tmp["score"] = 0.0
        top = tmp.nlargest(topn, "score")


        return top.merge(recipes_lookup, left_on="recipe_id", right_index=True, how="left")[
            ["recipe_id", "name", "minutes", "calories", "score"]
        ]
    if hist_len < 5:
        alpha_cf, beta_cb, gamma_kb = 0.3, 0.6, 0.1
    cf_raw = cf_model.predict_many(str(user_id), candidates)
    cf_norm = {rid: float(np.clip((float(s) - 1.0) / 4.0, 0.0, 1.0)) for rid, s in cf_raw.items()}
    liked = reviews_df[
        (reviews_df["user_id"].astype(str) == str(user_id)) &
        (reviews_df["rating"] >= 4)
    ]["recipe_id"].astype(str).tolist()
    cb_scores = {}
    if liked:
        liked_idx = [cb_model.recipe_ids.index(r) for r in liked if r in cb_model.recipe_ids]
        if liked_idx:
            user_vec = np.mean(cb_model.tfidf_matrix[liked_idx].toarray(), axis=0).reshape(1, -1)
            sims = cosine_similarity(user_vec, cb_model.tfidf_matrix).ravel()
            for rid, s in zip(cb_model.recipe_ids, sims):
                cb_scores[rid] = s
    fused = {}
    for rid in candidates:
        scf = np.clip(cf_norm.get(rid, 0.0), 0, 1)
        scb = np.clip(cb_scores.get(rid, 0.0), 0, 1)
        skb = np.clip(kb_bonus.get(rid, 0.0), 0, 1)
    


        # nutrition adjustments
        row = recipes_df.loc[recipes_df["recipe_id"].astype(str) == rid]
        protein_bonus = 0.0
        fat_penalty = 0.0

        if not row.empty:
            protein_val = row["protein"].values[0] if "protein" in row.columns else 0
            fat_val = row["fat"].values[0] if "fat" in row.columns else 0

            try:
                protein_bonus = np.tanh(float(protein_val) / 50) * (0.2 if prefs.get("min_protein", 0) > 0 else 0.1)
                fat_penalty = np.tanh(float(fat_val) / 30) * (0.2 if prefs.get("max_fat", 0) < 50 else 0.1)

            except Exception:
                pass

        fused[rid] = (
            alpha_cf * scf +
            beta_cb * scb +
            gamma_kb * skb +
            protein_bonus -
            fat_penalty
        )
    if fused:
        vals = np.array(list(fused.values()), dtype=float)
        min_s, max_s = vals.min(), vals.max()
        st.text(f"Score range before normalization: {min_s:.4f} — {max_s:.4f}")

        if max_s > min_s:
            exp_vals = np.exp((vals - np.mean(vals)) / (np.std(vals) + 1e-8))
            softmax_vals = exp_vals / np.sum(exp_vals)
            norm_vals = 0.1 + 0.8 * softmax_vals / (softmax_vals.max() + 1e-8)
            fused = dict(zip(fused.keys(), norm_vals))
        else:
            fused = {k: 0.5 for k in fused}




    st.text(f"Sample normalized (first 5): {list(fused.items())[:5]}")

    top_pairs = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:topn]
    top_df = pd.DataFrame(top_pairs, columns=["recipe_id", "normalized_score"])
    top_df = top_df.merge(recipes_lookup, left_on="recipe_id", right_index=True, how="left")

    top_df["normalized_score"] = top_df["normalized_score"].round(4)

    cols_to_show = [c for c in ["recipe_id", "name", "minutes", "calories"] if c in top_df.columns]
    return top_df[cols_to_show].copy()

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(page_title="Hybrid Food Recipe Recommender", layout="wide")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/857/857681.png", width=80)
st.sidebar.title("Recipe Recommender Dashboard")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fa; 
        color: #000000; 
    }
    [data-testid="stSidebar"] {
        background-color: #1E1E2F; 
        color: white;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }
    h1, h2, h3 {
        color: #2b3a67; 
    }
    div.stButton > button:first-child {
        background-color: #2b3a67;
        color: white;
        border: none;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }

    div.stButton > button:hover {
        background-color: #3e4a84;
        color: white;
        transform: scale(1.02);
    }
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.selectbox(
    "📂 Choose Page",
    [
        "🏠 About",
        "🍽️ Recommender",
        "👤 User Insights",
        "📈 Analytics & Visualizations",
        "⚙️ Data Overview",
        "💾 System Info"
    ]
)
st.sidebar.markdown("---")

if page == "🏠 About":
    st.title("About the Project")

    st.markdown("""
    ## Overview
    The **Hybrid Food Recipe Recommender System** provides intelligent and personalized meal suggestions by integrating **Collaborative Filtering (CF)**, **Content-Based Filtering (CB)**, and **Knowledge-Based Filtering (KB)**.  
    It combines user behavior, recipe semantics, and nutritional constraints to deliver both relevant and health-conscious recommendations.

    ## System Components
    - **Collaborative Filtering (CF):** Learns user–recipe relationships using cosine-based similarity of rating patterns to capture shared taste preferences.  
    - **Content-Based Filtering (CB):** Uses TF-IDF text embeddings of recipe ingredients, tags, and descriptions to identify semantically similar recipes.  
    - **Knowledge-Based Filtering (KB):** Applies user-defined constraints such as maximum calories, cooking time, and preferred cuisine to filter suitable options.  
    - **Hybrid Fusion:** Merges CF, CB, and KB scores using a weighted aggregation (e.g., *0.6×CF + 0.3×CB + 0.1×KB*) to balance personalization, similarity, and nutritional suitability.  
    - **Popularity Baseline:** Provides fallback recommendations for new users with limited rating history (cold-start scenario).  

    ## Methodology
    1. **Data Loading:** Retrieve recipes and reviews from a MySQL database using SQLAlchemy for efficient access and analysis.  
    2. **Preprocessing:** Clean, standardize, and enrich data with nutrition, tags, and text-based metadata.  
    3. **Model Training:** Build CF and CB models and integrate KB filtering for personalized constraints.  
    4. **Hybrid Scoring:** Compute final scores using weighted fusion of CF, CB, and KB outputs.  
    5. **Evaluation:** Use Precision@K, Recall@K, and F1-score to assess recommendation quality.  

    ## Technologies Used
    - **Programming:** Python (3.10+)  
    - **Framework:** Streamlit (for interactive UI)  
    - **Libraries:** Pandas · NumPy · Scikit-Learn · Matplotlib · Seaborn · WordCloud  
    - **Database:** MySQL (for recipe and review storage, accessed via SQLAlchemy)  
    - **Platform:** Windows 10 / Jupyter Notebook / Visual Studio Code  

    ## Objectives
    - Deliver accurate, personalized, and nutrition-aware recipe recommendations.  
    - Balance user preferences, health goals, and semantic relevance through hybrid modeling.  
    - Enable interactive exploration via a dynamic Streamlit dashboard.  
    - Address cold-start challenges with a popularity-based fallback mechanism.
    """)


elif page == "⚙️ Data Overview":
    st.title("Data Overview")
    st.subheader("Recipes Dataset")
    st.dataframe(recipes_used.head(10), use_container_width=True)
    st.write(f"Total recipes: {len(recipes_used)}")
    st.subheader("Reviews Dataset")
    st.dataframe(train.head(10), use_container_width=True)
    st.write(f"Total reviews: {len(train)}")
    st.markdown("Use these previews to validate data integrity before retraining.")

elif page == "🍽️ Recommender":
    st.title("🍽️ Hybrid Recipe Recommender (CF + CB + KB)")
    user_id = st.text_input("Enter User ID (e.g., 12345):", "")
    col1, col2, col3, col4, col5 = st.columns(5)
    max_minutes = col1.number_input("⏱️ Max Minutes", 0, 600, 45)
    max_calories = col2.number_input("🔥 Max Calories", 0, 2000, 800)
    min_protein = col3.number_input("💪 Min Protein (g)", 0, 100, 10)
    max_fat = col4.number_input("🥑 Max Fat (g)", 0, 100, 20)
    cuisine = col5.text_input("🍝 Cuisine Keyword", "italian")
    topn = col4.slider("🔢 Top N Recommendations", 5, 30, 10)
    if st.button("🔍 Generate Recommendations", use_container_width=True):
        if not user_id.strip():
            st.warning("Please enter a valid User ID.")
        else:
            prefs = {
    "max_minutes": max_minutes,
    "max_calories": max_calories,
    "min_protein": min_protein,
    "max_fat": max_fat,
    "cuisine": cuisine
}

            st.info("Generating hybrid recommendations... Please wait ⏳")

            df = recommend_cf_cb_kb(
                cf_model=cf_model,
                cb_model=cb_model,
                recipes_df=recipes_used,
                reviews_df=train,
                user_id=user_id.strip(),
                prefs=prefs,
                topn=topn
            )

            if df.empty:
               
                filtered_df, _ = apply_kb(recipes_used, prefs)
                
                if filtered_df.empty:
                    st.error("No recipes match your constraints. Try relaxing calorie, fat, or time limits.")
                else:
                    fallback_recipes = filtered_df.sample(min(10, len(filtered_df)), random_state=np.random.randint(0, 10000))
                    st.info("🔀 Showing 10 random recipes that match your nutritional & cuisine constraints:")
                    
                    cols_to_show = [c for c in ["recipe_id", "name", "minutes", "calories", "protein", "fat"] if c in fallback_recipes.columns]
                    st.dataframe(fallback_recipes[cols_to_show].reset_index(drop=True), use_container_width=True)
            else:
                st.success(f"✅ Top {len(df)} recommendations for User {user_id}:")
                st.dataframe(df, use_container_width=True)
                    


elif page == "👤 User Insights":
    st.title("👤 User Activity & Insights")

    st.subheader("Most Active Users")
    top_users = train["user_id"].value_counts().head(10)  # top 10 by count
    top_users = top_users.sort_values(ascending=False)    # ensure descending order
    st.bar_chart(top_users)
    st.subheader("Average Rating per Active User")
    avg_user_rating = (
        train[train["user_id"].isin(top_users.index)]
        .groupby("user_id")["rating"]
        .mean()
        .loc[top_users.index]  # preserve same order as top_users
    )
    st.line_chart(avg_user_rating)


    st.markdown("🔹 Tip: These insights help tune the recommender for highly active users.")
elif page == "📈 Analytics & Visualizations":
    st.title("Data Visualizations and Insights")
    st.markdown("""
    This section presents key analytical insights derived from the recipe and rating datasets.  
    The following visualizations summarize user behavior, recipe characteristics, and relationships among nutritional and temporal attributes.
    """)

    def show_fig():
        st.pyplot(plt.gcf())
        plt.clf()
    st.subheader("1. Ratings Distribution")
    st.markdown("""
    This plot illustrates how users rate recipes overall.  
    It helps identify whether the dataset is **biased toward high ratings** (common in recipe platforms) or if users provide balanced feedback across the 1–5 scale.
    """)
    plt.figure(figsize=(6,4))
    plt.hist(train["rating"], bins=5, color="skyblue", edgecolor="black")
    plt.title("Distribution of Recipe Ratings")
    plt.xlabel("Rating"); plt.ylabel("Number of Ratings")
    show_fig()
    st.subheader("2. Average Rating vs Cooking Time")
    st.markdown("""
    This line chart compares **average user satisfaction** (rating) against **recipe preparation time**.  
    It typically reveals whether users tend to prefer **quick, convenient recipes** or **longer, elaborate dishes**.
    """)
    mm = train.merge(recipes_used[["recipe_id", "minutes"]], on="recipe_id", how="inner")
    mm = mm[mm["minutes"].between(0, 180)]
    g = mm.groupby("minutes")["rating"].mean().reset_index()
    plt.figure(figsize=(8,4))
    plt.plot(g["minutes"], g["rating"], color="green")
    plt.title("Average Rating vs. Cooking Time")
    plt.xlabel("Cooking Time (minutes)"); plt.ylabel("Average Rating")
    show_fig()
    st.subheader("3. Top 20 Ingredients")
    st.markdown("""
    Displays the **most frequently used ingredients** across all recipes.  
    This provides insight into **common culinary components** and potential ingredient popularity trends.
    """)
    all_ings = [i for lst in recipes_used["ingredients"] if isinstance(lst, list) for i in lst]
    top_ing = pd.Series(all_ings).value_counts().head(20)
    plt.figure(figsize=(10,4))
    plt.barh(top_ing.index[::-1], top_ing.values[::-1], color="orange")
    plt.title("Most Frequently Used Ingredients")
    plt.xlabel("Occurrence Count"); plt.ylabel("Ingredient")
    show_fig()

    st.subheader("4. Calories vs Cooking Time (Colored by Rating)")
    st.markdown("""
    A scatter plot showing how **calorie content** relates to **cooking time**, with color intensity representing the **average rating**.  
    This visualization highlights trends such as whether **longer cooking times lead to higher calorie recipes** and if that affects user satisfaction.
    """)
    m2 = mm.merge(recipes_used[["recipe_id", "calories"]], on="recipe_id", how="left")
    plt.figure(figsize=(7,5))
    sc = plt.scatter(m2["minutes"], m2["calories"], c=m2["rating"], cmap="viridis", alpha=0.6)
    plt.colorbar(sc, label="Average Rating")
    plt.title("Calories vs. Cooking Time (Color = Rating)")
    plt.xlabel("Cooking Time (minutes)"); plt.ylabel("Calories")
    show_fig()
    st.subheader("5. Correlation Heatmap (Nutrition Facts)")
    st.markdown("""
    The correlation heatmap captures the **relationships among nutritional attributes**, such as calories, fat, sugar, and protein.  
    Strong positive or negative correlations reveal underlying **nutritional dependencies** — for instance, high calorie items often correlate with high fat and carbohydrate content.
    """)
    nut_cols = ["calories","fat","sat_fat","sodium","carbs","sugar","protein"]
    nutr = recipes_used[nut_cols].dropna()
    corr = nutr.corr().values
    plt.figure(figsize=(8,6))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Correlation Coefficient")
    plt.xticks(range(len(nut_cols)), nut_cols, rotation=45)
    plt.yticks(range(len(nut_cols)), nut_cols)
    plt.title("Correlation Heatmap of Nutritional Attributes")
    show_fig()

elif page == "💾 System Info":
    import platform, sys, sklearn
    st.title("System & Model Information")

    st.markdown("### Environment")
    st.json({
        "Python Version": sys.version.split()[0],
        "Platform": platform.system(),
        "Pandas": pd.__version__,
        "NumPy": np.__version__,
        "Scikit-Learn": sklearn.__version__,
        "Matplotlib": plt.matplotlib.__version__,
    })

    st.markdown("### Model Info")
    st.json({
        "Model Type": "User-Based Collaborative Filtering + Knowledge-Based Filter",
        "Neighbors (k)": cf_model.k,
        "Shrinkage": cf_model.shrinkage,
        "Center": cf_model.center,
        "Train Size": len(train),
        "Test Size": len(test)
    })

    st.success("System information loaded successfully!")

