import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    anime = pd.read_csv("data/anime-filtered.csv")
    rating = pd.read_csv("data/final_animedataset.csv")
    user = pd.read_csv("data/users-score-2023.csv")
    merged = rating.merge(anime, on="anime_id").merge(user, on="user_id")
    return anime, rating, user, merged

anime, rating, user, merged_df = load_data()

st.title("Anime Recommendation & Analytics App")

# Tabs for navigation
tab1, tab2 = st.tabs(["🔍 Recommend", "📊 Visualize"])

with tab1:
    st.header("Anime Recommender")

    user_id = st.selectbox("Select User ID", merged_df["user_id"].unique())
    top_n = st.slider("Number of Recommendations", 1, 10, 5)

    st.write("⚠️ You can plug in your actual model here to generate recommendations.")
    
    # TEMP: dummy recommendations
    user_history = merged_df[merged_df["user_id"] == user_id]
    already_seen = set(user_history["anime_id"])
    unseen_anime = anime[~anime["anime_id"].isin(already_seen)]
    recs = unseen_anime.sample(top_n)

    st.subheader("Recommended Anime")
    for _, row in recs.iterrows():
        st.markdown(f"**{row['name']}** - {row['genre']}")

with tab2:
    st.header("Visualize Anime Ratings")

    viz_choice = st.selectbox("Choose a Visualization", [
        "Rating Distribution by Gender",
        "Top 5 Most Popular Anime by Gender",
        "Average Score by Source",
        "Rating vs. Popularity",
        "User Rating Consistency"
    ])

    if viz_choice == "Rating Distribution by Gender":
        fig = plt.figure()
        sns.histplot(data=merged_df, x="score", hue="gender", multiple="stack", bins=10)
        plt.title("Rating Distribution by Gender")
        st.pyplot(fig)

    elif viz_choice == "Top 5 Most Popular Anime by Gender":
        fig = plt.figure()
        top_anime_by_gender = merged_df.groupby(['gender', 'name'])['score'].count().reset_index()
        top_anime_by_gender = top_anime_by_gender.sort_values(['gender', 'score'], ascending=False)
        genders = top_anime_by_gender['gender'].unique()
        for g in genders:
            subset = top_anime_by_gender[top_anime_by_gender['gender'] == g].head(5)
            sns.barplot(data=subset, y="name", x="score")
            plt.title(f"Top 5 Most Rated Anime - {g}")
            st.pyplot(plt.gcf())
            plt.clf()

    elif viz_choice == "Average Score by Source":
        fig = plt.figure()
        source_scores = merged_df.groupby("source")["score"].mean().sort_values(ascending=False)
        sns.barplot(x=source_scores.values, y=source_scores.index)
        plt.title("Average Rating by Anime Source")
        st.pyplot(fig)

    elif viz_choice == "Rating vs. Popularity":
        fig = plt.figure()
        anime_stats = merged_df.groupby("anime_id").agg({
            "score": "mean",
            "user_id": "count"
        }).rename(columns={"score": "avg_score", "user_id": "num_ratings"})
        sns.scatterplot(data=anime_stats, x="num_ratings", y="avg_score")
        plt.title("Rating vs. Popularity")
        plt.xscale("log")
        st.pyplot(fig)

    elif viz_choice == "User Rating Consistency":
        fig = plt.figure()
        user_consistency = merged_df.groupby("user_id")["score"].std().dropna()
        sns.histplot(user_consistency, bins=30)
        plt.title("User Rating Consistency (Std Dev of Scores)")
        st.pyplot(fig)