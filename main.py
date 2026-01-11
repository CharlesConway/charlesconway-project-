#main.py file: 
import pandas as pd
from src.data_loader import FEATURES_CSV, TRACKS_CSV, TARGET, SEED, \
                            TABLES_DIR, FIGURES_DIR, AUDIO_COLS, GENRE_COL, \
                            EXPLICIT_COL, \
                            load_features, load_tracks, build_df_g, \
                            build_df_ey, df_g_no_genre, df_ey_no_exp_year

from src.models import make_rf_numeric, make_rf_with_genre, make_ridge_df_ey, make_lasso_df_ey, make_mlp_df_ey
from src.evaluation import train_eval, plot_rf_importance_with_ohe, \
                            plot_popularity_histogram, plot_correlation_matrix

#Create output folders if they do not exist: 
def ensure_dirs():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()

    #Data loading (raw files renamed):
    df_features_raw = load_features(FEATURES_CSV)
    df_tracks_raw = load_tracks(TRACKS_CSV)
    print("Data Imported")
    #Build modeling datasets:
    df_g = build_df_g(df_features_raw)
    df_ey = build_df_ey(df_tracks_raw)

    plot_popularity_histogram(df_g,
                              title="df_g popularity",
                              outpath=FIGURES_DIR / "pop_hist_df_g.png",)
    plot_popularity_histogram(df_ey,
                              title="df_ey popularity",
                              outpath=FIGURES_DIR / "pop_hist_df_ey.png",)

    print("Modeling datasets created")
    #Ablation modeling datasets (for comparison)
    df_g_ng = df_g_no_genre(df_g)
    df_ey_no = df_ey_no_exp_year(df_ey)
    print("Ablated Modeling Datasets created")
    results = []

    plot_correlation_matrix(dataset=df_g_ng, 
                            title="Correlation Matrix df_g",
                            outpath=FIGURES_DIR / "corr_mat_df_g.png")
    plot_correlation_matrix(dataset=df_ey_no, 
                            title="Correlation Matrix df_ey_no",
                            outpath=FIGURES_DIR / "corr_mat_df_ey_no.png")

    # #Modelling df_g (Random Forest with one-hot encoded genre): 
    y = df_g[TARGET]
    X = df_g.drop(columns=[TARGET])
    numeric_cols_g = [c for c in AUDIO_COLS if c in X.columns] 
    print("Estimating Random Forest on df_g")
    rf_g = make_rf_with_genre(numeric_cols=numeric_cols_g)
    res = train_eval(rf_g, X, y, seed=SEED)
    results.append({"dataset":"df_g", "model":"RF_with_genre", **res})
    print("Estimation done")
    #Modelling df_g Ablated (no genre) (Random Forest with numerics and binary cat)
    y = df_g_ng[TARGET]
    X = df_g_ng.drop(columns=[TARGET])
    print("Estimating Random Forest on Ablated df_g")
    rf_g_ng = make_rf_numeric(numeric_cols=X.columns)
    res = train_eval(rf_g_ng, X, y, seed=SEED)
    results.append({"dataset":"df_g_no_genre", "model":"RF", **res})
    print("Estimation done")
    # Modelling df_ey full: Calling all model functions to evaluate RF + Ridge + Lasso + MLP
    y = df_ey[TARGET]
    X = df_ey.drop(columns=[TARGET])
    print("Estimating Random Forest on df_ey")
    rf_ey = make_rf_numeric(numeric_cols=X.columns)
    res = train_eval(rf_ey, X, y, seed=SEED)
    results.append({"dataset":"df_ey", "model":"RF", **res})
    print("Estimation done")

    plot_rf_importance_with_ohe(
        rf_ey,
        numeric_cols=numeric_cols_g,
        categorical_cols=None,
        title="RF feature importance – df_ey (with explicit and year)",
        outpath=FIGURES_DIR / "rf_importance_df_ey.png",
        top_n=15
    )

    numeric_cols_ey = [c for c in X.columns if c != "explicit"]
    print("Estimating Ridge on df_ey")
    ridge = make_ridge_df_ey(numeric_cols=numeric_cols_ey)
    print("Estimation done")
    print("Estimating Lasso on df_ey")
    lasso = make_lasso_df_ey(numeric_cols=numeric_cols_ey)
    print("Estimation done")
    print("Estimating MLP on df_ey")
    mlp   = make_mlp_df_ey(numeric_cols=numeric_cols_ey)
    print("Estimation done")

    for name, model in [("Ridge", ridge), ("Lasso", lasso), ("MLP", mlp)]:
        res = train_eval(model, X, y, seed=SEED)
        results.append({"dataset":"df_ey", "model":name, **res})

    #Modelling df_ey ablated (no explicit+release_year) (Randon Forest with numerics)
    y = df_ey_no[TARGET]
    X = df_ey_no.drop(columns=[TARGET])
    print("Estimating Random Forest on Ablated df_ey")
    rf_ey_no = make_rf_numeric(numeric_cols=X.columns)
    res = train_eval(rf_ey_no, X, y, seed=SEED)
    results.append({"dataset":"df_ey_no_exp_year", "model":"RF", **res})
    print("Estimation done")

    #Save results table to results file for report: 
    print("Saving results to metrics")
    results_df = pd.DataFrame(results).sort_values(["dataset","model"])
    out_csv = TABLES_DIR / "metrics.csv"
    results_df.to_csv(out_csv, index=False)
    print(results_df)
    print(f"\nSaved metrics to: {out_csv}")

    #Plot (Random Forest with genre importance)
    #Done on the full (all of the observations) dataset just for importance plot: 

    rf_g.fit(df_g.drop(columns=[TARGET]), df_g[TARGET])

    plot_rf_importance_with_ohe(
        rf_g,
        numeric_cols=numeric_cols_g,
        categorical_cols=[GENRE_COL],
        title="RF feature importance – df_g (with genre)",
        outpath=FIGURES_DIR / "rf_importance_df_g.png",
        top_n=15
    )

    return

if __name__ == "__main__":

    main()

