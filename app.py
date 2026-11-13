
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved files
df = pd.read_csv("spotify_songs_clustered.csv")
kmeans_model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
X_scaled = np.load("scaled_audio_features.npy")

# Page settings
st.set_page_config(
    page_title="Spotify Song Recommendation",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify Song Recommendation System")

st.write(
    "Discover songs similar to your favourite Spotify tracks "
    "using Machine Learning and K-Means Clustering."
)

# Song selection
song_list = sorted(
    df["track_name"].dropna().unique().tolist()
)

selected_song = st.selectbox(
    "Choose a song",
    song_list
)

if st.button("Get Recommendations"):

    matches = df[
        df["track_name"] == selected_song
    ]

    chosen_pos = matches.index[0]
    chosen = df.iloc[chosen_pos]

    st.subheader("Selected Song")

    st.write("**Song:**", chosen["track_name"])
    st.write("**Artist:**", chosen["track_artist"])
    st.write("**Genre:**", chosen["playlist_genre"])

    # Find songs in the same cluster
    cluster = chosen["cluster"]

    candidates = np.where(
        df["cluster"].to_numpy() == cluster
    )[0]

    # Remove selected song and duplicate copies
    candidates = candidates[candidates != chosen_pos]

    candidates = candidates[
        df.iloc[candidates]["track_id"].to_numpy()
        != chosen["track_id"]
    ]

    # Calculate distances
    distances = np.linalg.norm(
        X_scaled[candidates] - X_scaled[chosen_pos],
        axis=1
    )

    # Find closest songs
    best = candidates[np.argsort(distances)[:10]]

    recommendations = df.iloc[best][
        [
            "track_name",
            "track_artist",
            "playlist_genre",
            "track_popularity"
        ]
    ].reset_index(drop=True)

    recommendations.index += 1

    st.subheader("🎧 Recommended Songs")
    st.dataframe(recommendations, use_container_width=True)

    st.success("Recommendations generated successfully!")
