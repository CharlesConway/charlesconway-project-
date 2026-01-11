# src/evaluation.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Create evaluation function (for dependant y and independant X training on 80% and testing on 20%): 
def train_eval(model, X, y, test_size=0.2, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    #print(X.columns)
    #print(X.head(5))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 100)

    return {
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
#Plot feature importance (one of my main plots for report): 
def plot_rf_importance_with_ohe(model_pipeline, numeric_cols, title,categorical_cols=None, outpath=None, top_n=15):

    rf = model_pipeline.named_steps["rf"]
    pre = model_pipeline.named_steps["preprocess"]

    feature_names = list(numeric_cols)
    if categorical_cols is not None:
        ohe = pre.named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(categorical_cols)
        feature_names += list(cat_names)
    else:
        pre = model_pipeline.named_steps["preprocess"]
        feature_names = pre.get_feature_names_out()

    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)


    ax = importances.head(top_n).sort_values().plot(kind="barh")
    ax.set_xlabel("Feature importance (RF)")
    ax.set_title(title)
    plt.tight_layout()


    if outpath is not None:
        plt.savefig(outpath, dpi=200)

    return

def plot_popularity_histogram(dataset: pd.DataFrame, title, outpath = None):

    ax = dataset["popularity"].plot(kind="hist", orientation="vertical", bins=100)
    ax.set_xlabel("Popularity")
    ax.set_title(title)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=200)
    plt.close()

def plot_correlation_matrix(dataset: pd.DataFrame, title, outpath = None):
    corr = dataset.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title(title)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=200)
    plt.close()


