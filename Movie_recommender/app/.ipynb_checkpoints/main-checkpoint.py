from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd


app = FastAPI(
    title="Movie Recommendation API",
    version="1.0.0"
)


# -----------------------------
# Load files
# -----------------------------

movies = pickle.load(
    open("models/movies.pkl", "rb")
)

tfidf_matrix = pickle.load(
    open("models/tfidf_matrix.pkl", "rb")
)

nn_model = pickle.load(
    open("models/nn_model.pkl", "rb")
)

df = pd.read_pickle(
    "models/df.pkl"
)


# -----------------------------
# Request model
# -----------------------------

class MovieRequest(BaseModel):

    title: str
    n: int = 10


# -----------------------------
# Home
# -----------------------------

@app.get("/")
def home():

    return {
        "message": "Movie Recommendation API is running"
    }


# -----------------------------
# Health
# -----------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "movies_loaded": len(df)
    }


# -----------------------------
# Recommendation
# -----------------------------

def recommend_movies(title, n=10):

    title = title.lower().strip()

    if title not in movies:
        return None

    idx = movies[title]

    distances, indices = nn_model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=n + 1
    )

    similar_idx = indices[0][1:]

    recommendations = []

    for movie_idx, distance in zip(
        similar_idx,
        distances[0][1:]
    ):

        recommendations.append({

            "title": df["title"].iloc[movie_idx],

            "similarity_score": round(
                1 - float(distance),
                4
            )

        })

    return recommendations


# -----------------------------
# API endpoint
# -----------------------------

@app.post("/recommend")
def recommend(request: MovieRequest):

    if request.n < 1 or request.n > 50:

        raise HTTPException(
            status_code=400,
            detail="n must be between 1 and 50"
        )

    result = recommend_movies(
        request.title,
        request.n
    )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail=f"Movie '{request.title}' not found"
        )

    return {

        "input_movie": request.title,

        "recommendations": result
    }