# src/data_loader.py
import pandas as pd
import unicodedata
from pathlib import Path

#Here I define the paths to my files: 
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

#Define the data files: 
FEATURES_CSV = DATA_DIR / "SpotifyFeatures.csv"
TRACKS_CSV   = DATA_DIR / "tracks.csv"

#ML configuration: 
SEED = 42
TARGET = "popularity"

#What are the explicit columns I used:
GENRE_COL = "genre"
EXPLICIT_COL = "explicit"
RELEASE_DATE_COL = "release_date"
RELEASE_YEAR_COL = "release_year"

#State the audio features of interest (same as in report without popularity): 
AUDIO_COLS = [
    "acousticness","danceability","duration_ms","energy","instrumentalness",
    "key","liveness","loudness","mode","speechiness","tempo","time_signature","valence"
]

#Data Loader functions - Included here for a more condensed main.py: 
def load_features(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def load_tracks(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

#Remove the "´" Problem with Children's Music in genre (Briefly mentioned in Report, talked about in video): 
def clean_genre(series: pd.Series) -> pd.Series:
    # Normalize unicode (fix curly quotes etc.), strip spaces
    s = series.astype(str).apply(lambda x: unicodedata.normalize("NFKC", x))
    s = s.str.strip().str.replace(r"\s+", " ", regex=True)
    return s

def extract_release_year(series: pd.Series) -> pd.Series:
    #Extract first 4 digits (e.g. 2019) as release date is "YYYY-MM-DD" or "YYYY-MM" or "YYYY"
    year = (
        series.astype(str)
        .str.strip()
        .str.extract(r"^(\d{4})")[0]
        .astype("Int64")
    )
    return year

#Create Modelling dataset df_g (with data prepocessing): 
def build_df_g(df_features_raw: pd.DataFrame) -> pd.DataFrame:
    df_features_raw["mode"] = df_features_raw["mode"].map({"Major": 1, "Minor": 0})

    df_features_raw["key"] = \
        df_features_raw["key"].map({"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, \
                                    "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, \
                                    "A#": 10, "B": 11 })

    df_features_raw["time_signature"] = \
        df_features_raw["time_signature"].map({"5/4": 5, "4/4": 4, "3/4": 3, \
                                               "2/4": 2, "1/4": 1, "0/4": 0})
    cols = [TARGET] + AUDIO_COLS + [GENRE_COL]
    df_g = df_features_raw[cols].copy()
    df_g[GENRE_COL] = clean_genre(df_g[GENRE_COL])
    return df_g

#Create modelling dataset df_ey: 
def build_df_ey(df_tracks_raw: pd.DataFrame) -> pd.DataFrame:
    cols = [TARGET] + AUDIO_COLS + [EXPLICIT_COL, RELEASE_DATE_COL]
    df_ey = df_tracks_raw[cols].copy()
    #Convert release_date to release_year with preprocessing function extract_release_year(): 
    df_ey[RELEASE_YEAR_COL] = extract_release_year(df_ey[RELEASE_DATE_COL])
    df_ey = df_ey.drop(columns=[RELEASE_DATE_COL])
    return df_ey

#Create Modelling Ablated modelling datasets (for comparison): 
def df_g_no_genre(df_g: pd.DataFrame) -> pd.DataFrame:
    return df_g.drop(columns=[GENRE_COL]).copy()

def df_ey_no_exp_year(df_ey: pd.DataFrame) -> pd.DataFrame:
    return df_ey.drop(columns=[EXPLICIT_COL, RELEASE_YEAR_COL]).copy()
