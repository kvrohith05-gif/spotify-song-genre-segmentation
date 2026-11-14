# Spotify Song Recommendation System

A Machine Learning-based Spotify Song Recommendation System that recommends songs similar to a user's favourite track using **K-Means Clustering** and audio feature similarity.

The application allows users to select a Spotify song and discover similar songs based on their audio characteristics.

## Live Application

**Try the application here:**

[Spotify Song Recommendation System](https://spotify-song-genre-segmentation-ahyfnqj2liycpchgrkgzpr.streamlit.app/)

## Project Overview

Music recommendation systems help users discover songs based on their listening preferences.

This project uses Spotify song data and Machine Learning techniques to group songs with similar audio characteristics. When a user selects a song, the system recommends other songs that have similar audio features.

## Objectives

- Analyze Spotify song data.
- Perform data cleaning and preprocessing.
- Explore audio features through visualizations.
- Apply K-Means Clustering to group similar songs.
- Build a song recommendation system.
- Deploy the application using Streamlit.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- K-Means Clustering
- PCA
- Joblib
- Streamlit
- Google Colab
- GitHub

## Dataset

**Dataset:** Spotify 2019–2020 Songs Dataset

Source: [Spotify Songs Dataset](https://github.com/rfordatascience/tidytuesday/blob/master/data/2020/2020-01-21/spotify_songs.csv)

The dataset contains approximately 33,000 songs and includes information such as song name, artist, genre, popularity and audio features.

## Project Methodology

### 1. Data Collection
Collected the Spotify songs dataset containing song details and audio characteristics.

### 2. Data Preprocessing
- Checked missing values.
- Removed unnecessary columns.
- Selected relevant audio features.
- Standardized numerical features using StandardScaler.

### 3. Exploratory Data Analysis (EDA)
Performed visual analysis using:
- Genre distribution.
- Audio feature histograms.
- Popularity analysis.
- Correlation heatmap.
- Danceability and energy comparison.

### 4. K-Means Clustering
Applied K-Means Clustering to group songs based on their audio features.

Tested different cluster values and selected K = 2 based on the highest silhouette score among the tested values.

**Best Silhouette Score: 0.1462**

### 5. Dimensionality Reduction
Used Principal Component Analysis (PCA) to visualize the song clusters in a lower-dimensional space.

### 6. Song Recommendation
Developed a recommendation function using Euclidean distance to identify songs with similar audio characteristics within the same cluster.

### 7. Web Application
Created an interactive application using Streamlit, allowing users to select a song and receive similar song recommendations.

## Project Structure

```text
spotify-song-genre-segmentation/
│
├── app.py
├── requirements.txt
├── spotify_songs_clustered.csv
├── kmeans_model.pkl
├── scaler.pkl
├── scaled_audio_features.npy
├── README.md

```

## Run the Project Locally

**Step 1:** Clone the repository.

```bash
git clone https://github.com/kvrohith05-gif/spotify-song-genre-segmentation.git
```

**Step 2:** Open the project folder.

```bash
cd spotify-song-genre-segmentation
```

**Step 3:** Install the required libraries.

```bash
pip install -r requirements.txt
```

**Step 4:** Run the Streamlit application.

```bash
streamlit run app.py
```

## Google Colab Notebook

[Open Project Notebook]https://colab.research.google.com/github/kvrohith05-gif/spotify-song-genre-segmentation/blob/main/Spotify_Song_Recommendation_System.ipynb

## Future Enhancements

- Improve recommendation quality.
- Experiment with different clustering algorithms.
- Add personalized playlists.
- Include more Spotify audio features.
- Improve the user interface.

## Author

**K Venkata Rohith**

GitHub: [kvrohith05-gif](https://github.com/kvrohith05-gif)


