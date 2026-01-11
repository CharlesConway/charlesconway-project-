# src/models.py
#Import essential modules for my models: 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor

from .data_loader import GENRE_COL, EXPLICIT_COL, SEED

#Create function to run xRandom Forest with numerics function: 
def make_rf_numeric(numeric_cols):
    preprocess = ColumnTransformer([("num", "passthrough", numeric_cols)], remainder="drop")
    rf = RandomForestRegressor(
        #400 trees, minimum of two branchs at each decision node: 
        n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1
    )
    return Pipeline([("preprocess", preprocess), ("rf", rf)])

#Create function to run Random Forest model on df_g, same tree parameters as previous RF:  
def make_rf_with_genre(numeric_cols):
    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [GENRE_COL]),
        ],
        remainder="drop",
    )
    rf = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1
    )
    return Pipeline([("preprocess", preprocess), ("rf", rf)])

#Create Ridge regression function on df_ey:
def make_ridge_df_ey(numeric_cols):
    preprocess = ColumnTransformer(
        [
            #Needs to be normalized to run: 
            ("num", StandardScaler(), numeric_cols),
            ("bin", "passthrough", [EXPLICIT_COL]),
        ],
        remainder="drop",
    )
    ridge = Ridge(alpha=1.0, random_state=SEED)
    return Pipeline([("preprocess", preprocess), ("ridge", ridge)])

#Create Lasso regression function on df_ey: 
def make_lasso_df_ey(numeric_cols):
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("bin", "passthrough", [EXPLICIT_COL]),
        ],
        remainder="drop",
    )
    #Initial alpha set at 0.01, but changing it has minimal impact on the r2: 
    lasso = Lasso(alpha=0.01, max_iter=10000, random_state=SEED)
    return Pipeline([("preprocess", preprocess), ("lasso", lasso)])

#Create Multi Layered Perceptron regression: 
def make_mlp_df_ey(numeric_cols):
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("bin", "passthrough", [EXPLICIT_COL]),
        ],
        remainder="drop",
    )
    #Initial neuron values at 64, 32 but not much variation is present the deeper you go: 
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        random_state=SEED,
    )
    return Pipeline([("preprocess", preprocess), ("mlp", mlp)])
