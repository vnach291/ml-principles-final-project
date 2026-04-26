    #!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path
import copy
import gc
import datetime
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, clone, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.svm import SVC

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, XGBRegressor

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings("ignore")

# Strict fold synchronization for every Level 1 model.
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

TARGET_COL = "INDICATED_DAMAGE"

# ---------------------------------------------------------------------------
# Quick-test lever: set to e.g. 0.05 to run the full pipeline on 5% of data.
# Set back to 1.0 for production runs.
# ---------------------------------------------------------------------------
#DATA_FRACTION = 0.001
DATA_FRACTION = 1.0
GLOBAL_IMPUTATION = False  # If True, imputes full dataset before CV. If False, imputes per-fold.

TARGET_ENCODE_COLS = [
    "SPECIES_ID",
    "AIRPORT",
    "AIRCRAFT",
    "AMA",
    "AMO",
    "EMA",
    "EMO"
]

BASE_DIR = Path(".")
if not (BASE_DIR / "train_merg.csv").exists():
    BASE_DIR = Path("ml final project")

train_path = BASE_DIR / "train_merg.csv"
test_path = BASE_DIR / "test.csv"

train_df_raw = pd.read_csv(train_path)
test_df_raw = pd.read_csv(test_path) if test_path.exists() else None


# ---------------------------------------------------------------------------
# Extract raw REMARKS text *before* cleaning drops the column.
# This is used later by the TF-IDF text model.
# ---------------------------------------------------------------------------
def _normalize_columns_quick(df):
    out = df.copy()
    out.columns = [c.strip().upper() for c in out.columns]
    return out

_temp_train = _normalize_columns_quick(train_df_raw)
_remarks_train_all = (
    _temp_train["REMARKS"].fillna("").astype(str).str.strip()
    if "REMARKS" in _temp_train.columns
    else pd.Series("", index=train_df_raw.index)
)
_comments_train_all = (
    _temp_train["COMMENTS"].fillna("").astype(str).str.strip()
    if "COMMENTS" in _temp_train.columns
    else pd.Series("", index=train_df_raw.index)
)

if test_df_raw is not None:
    _temp_test = _normalize_columns_quick(test_df_raw)
    _remarks_test_all = (
        _temp_test["REMARKS"].fillna("").astype(str).str.strip()
        if "REMARKS" in _temp_test.columns
        else pd.Series("", index=test_df_raw.index)
    )
    _comments_test_all = (
        _temp_test["COMMENTS"].fillna("").astype(str).str.strip()
        if "COMMENTS" in _temp_test.columns
        else pd.Series("", index=test_df_raw.index)
    )
else:
    _remarks_test_all = None
    _comments_test_all = None

del _temp_train
if test_df_raw is not None:
    del _temp_test


# ---------------------------------------------------------------------------
# Cleaning functions (unchanged)
# ---------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().upper() for c in out.columns]
    return out


def infer_time_of_day(time_value):
    if pd.isna(time_value):
        return np.nan
    text = str(time_value).strip()
    if not text:
        return np.nan
    try:
        hour = int(text.split(":")[0])
    except Exception:
        return np.nan

    if 5 <= hour <= 6:
        return "Dawn"
    if 7 <= hour <= 17:
        return "Day"
    if 18 <= hour <= 19:
        return "Dusk"
    return "Night"

# def month_to_group(month_value):
#     try:
#         month = int(month_value)
#     except Exception:
#         return np.nan

#     if month in (12, 1, 2):
#         return "Winter"
#     if month in (3, 4, 5):
#         return "Spring"
#     if month in (6, 7, 8):
#         return "Summer"
#     if month in (9, 10, 11):
#         return "Fall"
#     return np.nan


def has_text_flag(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("").astype(int)


def clean_wildlife_strike_df(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    d = normalize_columns(df)

    # Fill TIME_OF_DAY from TIME when missing, then drop TIME.
    if "TIME_OF_DAY" in d.columns and "TIME" in d.columns:
        tod_missing = d["TIME_OF_DAY"].isna() | d["TIME_OF_DAY"].astype(str).str.strip().eq("")
        d.loc[tod_missing, "TIME_OF_DAY"] = d.loc[tod_missing, "TIME"].apply(infer_time_of_day)
    elif "TIME" in d.columns:
        d["TIME_OF_DAY"] = d["TIME"].apply(infer_time_of_day)

    # Encode free-text fields as missing-vs-not-missing indicators.
    if "LOCATION" in d.columns:
        d["LOCATION_HAS_TEXT"] = has_text_flag(d["LOCATION"])
    if "COMMENTS" in d.columns:
        d["COMMENTS_HAS_TEXT"] = has_text_flag(d["COMMENTS"])
    if "REMARKS" in d.columns:
        d["REMARKS_HAS_TEXT"] = has_text_flag(d["REMARKS"])

    # Convert INCIDENT_MONTH to grouped categorical buckets.
    # if "INCIDENT_MONTH" in d.columns:
    #     d["INCIDENT_MONTH_GROUP"] = d["INCIDENT_MONTH"].apply(month_to_group).astype("category")

    # AC_MASS is ordinal (1..5).
    if "AC_MASS" in d.columns:
        d["AC_MASS"] = pd.to_numeric(d["AC_MASS"], errors="coerce")

    # Force requested categorical fields into categorical dtype.
    for col in [
        "ENG_1_POS",
        "ENG_2_POS",
        "ENG_3_POS",
        "ENG_4_POS",
        "REMAINS_COLLECTED",
        "REMAINS_SENT",
        "TIME_OF_DAY",
        # "INCIDENT_MONTH_GROUP",
        "INCIDENT_MONTH"
    ]:
        if col in d.columns:
            d[col] = d[col].astype("string").astype("category")

    # Drop fields requested in notes.
    drop_cols = [
        "AIRPORT_ID",
        "LATITUDE",
        "LONGITUDE",
        "OPID",
        "REG",
        "FLT",
        # "AMA",
        # "AMO",
        # "EMA",
        # "EMO",
        "SPECIES",
        #"SPECIES_ID",
        "ENROUTE_STATE",
        "INDEX_NR",
        "TRANSFER",
        "TIME",
        "LOCATION",
        "COMMENTS",
        "REMARKS",
        # "INCIDENT_MONTH",
    ]
    d = d.drop(columns=[c for c in drop_cols if c in d.columns], errors="ignore")

    if target_col in d.columns:
        target = pd.to_numeric(d[target_col], errors="coerce")
        if target.isna().any():
            mapped = (
                d[target_col]
                .astype(str)
                .str.strip()
                .str.upper()
                .map(
                    {
                        "1": 1,
                        "0": 0,
                        "YES": 1,
                        "NO": 0,
                        "Y": 1,
                        "N": 0,
                        "TRUE": 1,
                        "FALSE": 0,
                        "DAMAGE": 1,
                        "NO DAMAGE": 0,
                    }
                )
            )
            target = target.fillna(mapped)
        d[target_col] = target.astype("Int64")

    return d


train_df = clean_wildlife_strike_df(train_df_raw)
test_df = clean_wildlife_strike_df(test_df_raw) if test_df_raw is not None else None

if TARGET_COL not in train_df.columns:
    raise ValueError(f"{TARGET_COL} not found in training data after cleaning.")

train_df = train_df.dropna(subset=[TARGET_COL]).copy()

# --- Subsample for quick testing ---
if DATA_FRACTION < 1.0:
    from sklearn.model_selection import train_test_split as _stratified_split
    _keep_idx, _ = _stratified_split(
        train_df.index, test_size=1.0 - DATA_FRACTION,
        stratify=train_df[TARGET_COL], random_state=42,
    )
    train_df = train_df.loc[_keep_idx].copy()
    print(f"⚠ DATA_FRACTION={DATA_FRACTION} → subsampled to {len(train_df):,} rows")

y = train_df[TARGET_COL].astype(int)
X = train_df.drop(columns=[TARGET_COL]).copy()

X_test = None
if test_df is not None:
    if TARGET_COL in test_df.columns:
        test_df = test_df.drop(columns=[TARGET_COL])
    X_test = test_df.copy()

if X_test is not None:
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = np.nan
    X_test = X_test[X.columns]

categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
# Exclude target-encoded columns from categorical processing so they get routed to numeric_pipe
categorical_cols = [c for c in categorical_cols if c not in TARGET_ENCODE_COLS]
numeric_cols = [c for c in X.columns if c not in categorical_cols]
X_native = X.copy()
X_test_native = X_test.copy() if X_test is not None else None

for col in categorical_cols:
    X_native[col] = X_native[col].astype("category")
    if X_test_native is not None:
        X_test_native[col] = X_test_native[col].astype("category")

neg_count = int((y == 0).sum())
pos_count = int((y == 1).sum())
scale_pos_weight = float(neg_count / max(pos_count, 1))
cat_class_weights = [1.0, scale_pos_weight]
MISSING_CAT_TOKEN = "__MISSING__"
IMPUTER_RANDOM_STATE = None
IMPUTER_N_NEAREST = max(5, min(20, len(numeric_cols) // 2)) if len(numeric_cols) > 5 else None
NO_DAMAGE_DROP_FRACTION = 0.001
UNDERSAMPLE_TRAIN_FOLDS = True
UNDERSAMPLE_SEED = 42

# Align REMARKS text to training/test indices
X_text_train = _remarks_train_all.loc[y.index]
X_text_test = _remarks_test_all if _remarks_test_all is not None else None

# Align COMMENTS text to training/test indices
X_text_comments_train = _comments_train_all.loc[y.index]
X_text_comments_test = _comments_test_all if _comments_test_all is not None else None


# ---------------------------------------------------------------------------
# Run directory & persistence settings
# ---------------------------------------------------------------------------
RUN_DIR = Path("runs") / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / "models").mkdir(exist_ok=True)
(RUN_DIR / "plots").mkdir(exist_ok=True)
SAVE_FOLD_MODELS = True          # set False to skip saving fitted models
SHOW_PLOTS = True                # set False for headless (save-only) runs

print(f"Run directory: {RUN_DIR}")

# ---------------------------------------------------------------------------
# Global Target Encoding for Inference
# ---------------------------------------------------------------------------
import joblib
global_te = TargetEncoder(random_state=42)
global_te.fit(X[TARGET_ENCODE_COLS].astype(str), y)
joblib.dump(global_te, RUN_DIR / "target_encoder.joblib")


# ---------------------------------------------------------------------------
# Pipeline builders — parameterized for feature subsets
# ---------------------------------------------------------------------------
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, keep_indices):
        self.keep_indices = keep_indices
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if hasattr(X, "iloc"):
            return X.iloc[:, self.keep_indices]
        return X[:, self.keep_indices]

def build_math_pipeline(estimator, num_cols=None, cat_cols=None, dense_output=False, all_num_cols=None):
    num_cols = num_cols if num_cols is not None else numeric_cols
    cat_cols = cat_cols if cat_cols is not None else categorical_cols
    all_num_cols = all_num_cols if all_num_cols is not None else numeric_cols

    keep_num_indices = [all_num_cols.index(c) for c in num_cols if c in all_num_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", IterativeImputer(
                estimator=XGBRegressor(          # ← strongest option inside sklearn
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1,
                    enable_categorical=False
                ),
                max_iter=15,                     # more iterations = better convergence
                random_state=42,
                sample_posterior=False,          # XGBRegressor has no predict(return_std)
                imputation_order="ascending"
            )),
            ("scaler", StandardScaler()),
            ("dropper", DropColumnsTransformer(keep_num_indices)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_CAT_TOKEN)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(all_num_cols)),
            ("cat", categorical_pipe, list(cat_cols)),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])


def build_tree_ohe_pipeline(estimator, num_cols=None, cat_cols=None, all_num_cols=None):
    num_cols = num_cols if num_cols is not None else numeric_cols
    cat_cols = cat_cols if cat_cols is not None else categorical_cols
    all_num_cols = all_num_cols if all_num_cols is not None else numeric_cols

    keep_num_indices = [all_num_cols.index(c) for c in num_cols if c in all_num_cols]

    numeric_pipe = Pipeline(
        steps=[
            (
                "imputer",
                IterativeImputer(
                    estimator=RandomForestRegressor(
                        n_estimators=80,
                        max_depth=8,
                        random_state=IMPUTER_RANDOM_STATE,
                        n_jobs=-1,
                    ),
                    max_iter=12,
                    initial_strategy="median",
                    n_nearest_features=IMPUTER_N_NEAREST,
                    imputation_order="random",
                    random_state=IMPUTER_RANDOM_STATE,
                ),
            ),
            ("dropper", DropColumnsTransformer(keep_num_indices)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_CAT_TOKEN)),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(all_num_cols)),
            ("cat", categorical_pipe, list(cat_cols)),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])


def first_grid_point(param_grid: dict) -> dict:
    return {k: (v[0] if isinstance(v, list) else v) for k, v in param_grid.items()}


def undersample_no_damage_rows(
    X_tr,
    y_tr: pd.Series,
    no_damage_drop_fraction: float = NO_DAMAGE_DROP_FRACTION,
    seed: int = UNDERSAMPLE_SEED,
):
    if not (0.0 <= no_damage_drop_fraction < 1.0):
        raise ValueError("no_damage_drop_fraction must be in [0, 1).")

    y_tr = y_tr.astype(int)
    pos_idx = y_tr[y_tr == 1].index
    neg_idx = y_tr[y_tr == 0].index

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    n_total = n_pos + n_neg

    if n_total == 0:
        return X_tr, y_tr, float("nan"), float("nan"), 0

    pos_rate_before = n_pos / n_total

    if n_neg == 0 or no_damage_drop_fraction == 0.0:
        return X_tr, y_tr, float(pos_rate_before), float(pos_rate_before), 0

    n_drop_neg = int(np.floor(n_neg * no_damage_drop_fraction))
    if n_drop_neg <= 0:
        return X_tr, y_tr, float(pos_rate_before), float(pos_rate_before), 0

    n_keep_neg = max(1, n_neg - n_drop_neg)
    rng = np.random.default_rng(seed)
    kept_neg_idx = rng.choice(np.array(neg_idx), size=n_keep_neg, replace=False)
    keep_idx = np.concatenate([np.array(pos_idx), kept_neg_idx])
    rng.shuffle(keep_idx)

    X_sub = X_tr.loc[keep_idx]
    y_sub = y_tr.loc[keep_idx].astype(int)

    pos_rate_after = float((y_sub == 1).mean())
    return X_sub, y_sub, float(pos_rate_before), pos_rate_after, int(n_drop_neg)


model_oof_predictions = {}
model_test_predictions = {}
model_cv_bal_acc = {}

level2_train_df = pd.DataFrame(index=X.index)
level2_test_df = pd.DataFrame(index=X_test.index) if X_test is not None else None


# ---------------------------------------------------------------------------
# run_level1_model — updated with model & plot saving
# ---------------------------------------------------------------------------
def _savefig(fig_or_plt, name: str):
    """Save figure to run directory and optionally show."""
    path = RUN_DIR / "plots" / f"{name}.png"
    fig_or_plt.savefig(path, dpi=100, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    # ALWAYS close to prevent memory leaks when generating thousands of plots
    if hasattr(fig_or_plt, "close"):
        fig_or_plt.close()
    else:
        plt.close("all")


def run_level1_model(
    model_name: str,
    estimator,
    param_grid: dict,
    X_data,
    y_data: pd.Series,
    use_grid_search: bool = False,
    fit_params: dict | None = None,
    X_holdout=None,
):
    fit_params = fit_params or {}
    oof_probs = np.zeros(len(X_data), dtype=float)
    holdout_probs = np.zeros(len(X_holdout), dtype=float) if X_holdout is not None else None
    fold_bal_acc = []
    fold_avg_precision = []
    fold_train_bal_acc = []
    fold_train_avg_precision = []
    chosen_params = first_grid_point(param_grid)
    overall_pos_rate = float(y_data.mean())
    strat_tol = max(0.002, 1.0 / max(len(y_data) // skf.n_splits, 1))

    def _predict_probs(fitted_model, X_eval):
        if hasattr(fitted_model, "predict_proba"):
            probs = fitted_model.predict_proba(X_eval)[:, 1]
        else:
            scores = fitted_model.decision_function(X_eval)
            probs = 1.0 / (1.0 + np.exp(-scores))

        if model_name == "knn":
            weight_for_no = 2.0
            weight_for_yes = 1.0
            p_yes = probs
            p_no = 1.0 - probs
            weighted_p_no = p_no * weight_for_no
            weighted_p_yes = p_yes * weight_for_yes
            probs = weighted_p_yes / (weighted_p_no + weighted_p_yes)

        return np.clip(probs, 1e-7, 1.0 - 1e-7)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_data, y_data), start=1):
        X_tr, X_val = X_data.iloc[tr_idx].copy(), X_data.iloc[val_idx].copy()
        y_tr, y_val = y_data.iloc[tr_idx], y_data.iloc[val_idx]

        X_holdout_cv = None
        if X_holdout is not None:
            X_holdout_cv = X_holdout.copy()

        if isinstance(X_tr, pd.DataFrame):
            enc_cols = [c for c in TARGET_ENCODE_COLS if c in X_tr.columns]
            if enc_cols:
                te_cv = TargetEncoder(random_state=42)
                train_nan_mask = X_tr[enc_cols].isna()
                val_nan_mask = X_val[enc_cols].isna()
                
                X_tr[enc_cols] = te_cv.fit_transform(X_tr[enc_cols].astype(str), y_tr)
                X_val[enc_cols] = te_cv.transform(X_val[enc_cols].astype(str))
                
                X_tr[enc_cols] = X_tr[enc_cols].mask(train_nan_mask)
                X_val[enc_cols] = X_val[enc_cols].mask(val_nan_mask)
                
                for col in enc_cols:
                    X_tr[col] = X_tr[col].astype("float64")
                    X_val[col] = X_val[col].astype("float64")
                    
                if X_holdout_cv is not None and all(c in X_holdout_cv.columns for c in enc_cols):
                    holdout_nan_mask = X_holdout_cv[enc_cols].isna()
                    X_holdout_cv[enc_cols] = te_cv.transform(X_holdout_cv[enc_cols].astype(str))
                    X_holdout_cv[enc_cols] = X_holdout_cv[enc_cols].mask(holdout_nan_mask)
                    for col in enc_cols:
                        X_holdout_cv[col] = X_holdout_cv[col].astype("float64")
        tr_pos_rate = float(y_tr.mean())
        val_pos_rate = float(y_val.mean())
        tr_neg_rate = 1.0 - tr_pos_rate
        val_neg_rate = 1.0 - val_pos_rate

        print(
            f"{model_name} | Fold {fold} stratification -> "
            f"train no/yes={tr_neg_rate:.4f}/{tr_pos_rate:.4f}, "
            f"val no/yes={val_neg_rate:.4f}/{val_pos_rate:.4f}, "
            f"global no/yes={(1.0 - overall_pos_rate):.4f}/{overall_pos_rate:.4f}"
        )

        if abs(tr_pos_rate - overall_pos_rate) > strat_tol or abs(val_pos_rate - overall_pos_rate) > strat_tol:
            raise ValueError(
                f"Stratification drift in fold {fold}. "
                f"train_pos={tr_pos_rate:.6f}, val_pos={val_pos_rate:.6f}, "
                f"global_pos={overall_pos_rate:.6f}, tol={strat_tol:.6f}"
            )

        X_fit, y_fit = X_tr, y_tr
        if UNDERSAMPLE_TRAIN_FOLDS:
            X_fit, y_fit, tr_pos_before, tr_pos_after, n_dropped = undersample_no_damage_rows(
                X_tr,
                y_tr,
                no_damage_drop_fraction=NO_DAMAGE_DROP_FRACTION,
                seed=UNDERSAMPLE_SEED + fold,
            )
            if n_dropped > 0:
                print(
                    f"{model_name} | Fold {fold} undersample -> "
                    f"train no/yes pre={(1.0 - tr_pos_before):.4f}/{tr_pos_before:.4f}, "
                    f"post={(1.0 - tr_pos_after):.4f}/{tr_pos_after:.4f}, "
                    f"dropped_no_damage_rows={n_dropped}"
                )

        fold_model = clone(estimator)
        fit_kwargs = dict(fit_params)
        
        # --- 1. ROBUST EARLY STOPPING CHECK ---
        model_params = fold_model.get_params() if hasattr(fold_model, "get_params") else {}
        has_es = (
            model_params.get("early_stopping_rounds") is not None
            or chosen_params.get("early_stopping_rounds") is not None
            or fit_kwargs.get("early_stopping_rounds") is not None
            or any("early_stopping" in str(cb).lower() for cb in fit_kwargs.get("callbacks", []))
        )
        
        if has_es:
            is_lgbm = type(fold_model).__name__.startswith("LGBM")
            
            if hasattr(fold_model, "steps"):
                est_name = fold_model.steps[-1][0]
                est_obj = fold_model.steps[-1][1]
                
                fit_kwargs.setdefault(f"{est_name}__eval_set", [(X_fit, y_fit), (X_val, y_val)])
                
                # Only silence if it's not LGBM in the pipeline
                if not type(est_obj).__name__.startswith("LGBM"):
                    fit_kwargs.setdefault(f"{est_name}__verbose", False)
            else:
                try:
                    import catboost
                    if isinstance(fold_model, catboost.CatBoostClassifier) or isinstance(fold_model, catboost.CatBoostRegressor):
                        fit_kwargs.setdefault("eval_set", [(X_val, y_val)])
                    else:
                        fit_kwargs.setdefault("eval_set", [(X_fit, y_fit), (X_val, y_val)])
                except ImportError:
                    fit_kwargs.setdefault("eval_set", [(X_fit, y_fit), (X_val, y_val)])
                
                # Only silence if it's not a direct LGBM 
                if not is_lgbm:
                    fit_kwargs.setdefault("verbose", False)
                else:
                    # Very new LightGBM requires passing a callback to silence the eval_set
                    try:
                        import lightgbm as lgb
                        cbs = fit_kwargs.get("callbacks", [])
                        cbs.append(lgb.log_evaluation(period=0))
                        fit_kwargs["callbacks"] = cbs
                    except Exception:
                        pass
                        
        if use_grid_search:
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid = GridSearchCV(
                estimator=fold_model,
                param_grid=param_grid,
                scoring="average_precision",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_fit, y_fit, **fit_kwargs)
            fitted_model = grid.best_estimator_
            chosen_params = grid.best_params_
        else:
            fitted_model = fold_model.set_params(**chosen_params)
            fitted_model.fit(X_fit, y_fit, **fit_kwargs)

        # --- SAVE FOLD MODEL ---
        if SAVE_FOLD_MODELS and RUN_DIR is not None:
            model_path = RUN_DIR / "models" / f"{model_name}_fold{fold}.joblib"
            try:
                joblib.dump(fitted_model, model_path)
            except Exception as e:
                print(f"  Warning: could not save model {model_name} fold {fold}: {e}")

        # --- 2. PLOT EVALUATIONS EVERY ITERATION ---
        evals_result = None
        if hasattr(fitted_model, "evals_result"):
            try:
                evals_result = fitted_model.evals_result()
            except Exception:
                pass
        elif hasattr(fitted_model, "evals_result_"):
            evals_result = fitted_model.evals_result_
        elif hasattr(fitted_model, "get_evals_result"):
            try:
                evals_result = fitted_model.get_evals_result()
            except Exception:
                pass
        
        if evals_result:
            all_metrics = set()
            for ds_name, m_dict in evals_result.items():
                if isinstance(m_dict, dict):
                    for m_name in m_dict.keys():
                        all_metrics.add(m_name)
            
            for m_name in all_metrics:
                plt.figure(figsize=(6, 3.5))
                for ds_name, m_dict in evals_result.items():
                    if isinstance(m_dict, dict) and m_name in m_dict:
                        plt.plot(m_dict[m_name], label=f"{ds_name}")
                plt.title(f"{model_name} Fold {fold} - {m_name}")
                plt.xlabel("Iterations")
                plt.ylabel(m_name)
                plt.legend()
                plt.tight_layout()
                _savefig(plt, f"{model_name}_fold{fold}_evals_{m_name}")

        train_probs = _predict_probs(fitted_model, X_fit)
        val_probs = _predict_probs(fitted_model, X_val)

        oof_probs[val_idx] = val_probs

        # --- FOLD THRESHOLD SWEEP ---
        best_fold_thresh = 0.5
        best_fold_bal_acc = 0.0
        
        # Test 50 thresholds between 0.1 and 0.9 to find the maximum possible Balanced Accuracy
        for thresh in np.linspace(0.1, 0.9, 50):
            temp_val_pred = (val_probs >= thresh).astype(int)
            temp_bal = balanced_accuracy_score(y_val, temp_val_pred)
            if temp_bal > best_fold_bal_acc:
                best_fold_bal_acc = temp_bal
                best_fold_thresh = thresh
                
        # Calculate final train/val metrics applying the very best threshold found
        train_pred = (train_probs >= best_fold_thresh).astype(int)
        val_pred = (val_probs >= best_fold_thresh).astype(int)
        
        train_bal = float(balanced_accuracy_score(y_fit, train_pred))
        val_bal = float(balanced_accuracy_score(y_val, val_pred))
        train_ap = float(average_precision_score(y_fit, train_probs))
        val_ap = float(average_precision_score(y_val, val_probs))
        train_loss = float(log_loss(y_fit, train_probs, labels=[0, 1]))
        val_loss = float(log_loss(y_val, val_probs, labels=[0, 1]))

        fold_train_bal_acc.append(train_bal)
        fold_bal_acc.append(val_bal)
        fold_train_avg_precision.append(train_ap)
        fold_avg_precision.append(val_ap)

        plt.figure(figsize=(6.0, 3.5))
        plt.plot(["train", "val"], [train_loss, val_loss], marker="o")
        plt.title(f"{model_name} | Fold {fold} loss")
        plt.ylabel("Log Loss")
        plt.tight_layout()
        _savefig(plt, f"{model_name}_fold{fold}_loss")

        if X_holdout is not None:
            fold_holdout_probs = _predict_probs(fitted_model, X_holdout_cv if X_holdout_cv is not None else X_holdout)
            holdout_probs += fold_holdout_probs / skf.n_splits

        print(
            f"{model_name} | Fold {fold} train/val -> "
            f"balanced_accuracy={train_bal:.4f}/{val_bal:.4f} (at best threshold {best_fold_thresh:.2f}), "
            f"average_precision={train_ap:.4f}/{val_ap:.4f}"
        )

    # --- OOF (OVERALL) THRESHOLD SWEEP ---
    best_overall_thresh = 0.5
    overall_bal_acc = 0.0
    
    for thresh in np.linspace(0.1, 0.9, 50):
        temp_oof_pred = (oof_probs >= thresh).astype(int)
        temp_oof_bal = balanced_accuracy_score(y_data, temp_oof_pred)
        if temp_oof_bal > overall_bal_acc:
            overall_bal_acc = temp_oof_bal
            best_overall_thresh = thresh
            
    overall_avg_precision = average_precision_score(y_data, oof_probs)

    folds_axis = np.arange(1, len(fold_bal_acc) + 1)
    plt.figure(figsize=(10, 3.8))
    plt.subplot(1, 2, 1)
    plt.plot(folds_axis, fold_train_bal_acc, marker="o", label="train")
    plt.plot(folds_axis, fold_bal_acc, marker="o", label="val")
    plt.title(f"{model_name} per-fold Balanced Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Balanced Accuracy")
    plt.xticks(folds_axis)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(folds_axis, fold_train_avg_precision, marker="o", label="train")
    plt.plot(folds_axis, fold_avg_precision, marker="o", label="val")
    plt.title(f"{model_name} per-fold Average Precision")
    plt.xlabel("Fold")
    plt.ylabel("Average Precision")
    plt.xticks(folds_axis)
    plt.legend()
    plt.tight_layout()
    _savefig(plt, f"{model_name}_per_fold_summary")

    model_oof_predictions[model_name] = oof_probs
    model_cv_bal_acc[model_name] = {
        "fold_balanced_accuracy": fold_bal_acc,
        "mean_balanced_accuracy": float(np.mean(fold_bal_acc)),
        "oof_balanced_accuracy": float(overall_bal_acc),
        "oof_best_threshold": float(best_overall_thresh),
        "fold_average_precision": fold_avg_precision,
        "mean_average_precision": float(np.mean(fold_avg_precision)),
        "oof_average_precision": float(overall_avg_precision),
        "chosen_params": {k: str(v) for k, v in chosen_params.items()},
    }

    if holdout_probs is not None:
        model_test_predictions[model_name] = holdout_probs

    level2_train_df[model_name] = oof_probs
    if level2_test_df is not None and holdout_probs is not None:
        level2_test_df[model_name] = holdout_probs

    print(f"{model_name} | OOF balanced_accuracy: {overall_bal_acc:.4f} (at overall optimal threshold {best_overall_thresh:.4f})")
    print(f"{model_name} | OOF average_precision: {overall_avg_precision:.4f}")

    return oof_probs, holdout_probs, model_cv_bal_acc[model_name]


cat_feature_indices = [X_native.columns.get_loc(c) for c in categorical_cols]

print(f"Train shape after cleaning: {X.shape}")
print(f"Test shape after cleaning: {None if X_test is None else X_test.shape}")
print(f"Class ratio (neg:pos) = {neg_count}:{pos_count} | scale_pos_weight = {scale_pos_weight:.3f}")


# In[3]:


# 1.5) Categorical type normalization for XGBoost/LightGBM/CatBoost native handling
MISSING_CAT_TOKEN = "__MISSING__"


def normalize_cat_series(series: pd.Series, missing_token: str = MISSING_CAT_TOKEN) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.fillna(missing_token)
    s = s.replace({"": missing_token, "<NA>": missing_token, "nan": missing_token, "None": missing_token})
    return s


category_mappings = {}

for col in categorical_cols:
    X[col] = normalize_cat_series(X[col])
    if X_test is not None:
        X_test[col] = normalize_cat_series(X_test[col])


X_native = X.copy()
X_test_native = X_test.copy() if X_test is not None else None

for col in categorical_cols:
    if X_test_native is not None:
        unified_categories = pd.Index(X_native[col].unique()).union(pd.Index(X_test_native[col].unique()))
    else:
        unified_categories = pd.Index(X_native[col].unique())

    # Ensure deterministic, list-based ordering for JSON saving
    stable_cats = list(unified_categories.dropna().sort_values())
    
    cat_type = pd.CategoricalDtype(categories=stable_cats, ordered=False)
    X_native[col] = X_native[col].astype(cat_type)
    if X_test_native is not None:
        X_test_native[col] = X_test_native[col].astype(cat_type)

    category_mappings[col] = stable_cats

# Save the exact categorical structures used to initialize the Boosters
with open(RUN_DIR / "category_mappings.json", "w") as f:
    json.dump(category_mappings, f, indent=2)

cat_feature_indices = [X_native.columns.get_loc(c) for c in categorical_cols]

print("Categorical normalization complete.")
print(f"Total categorical columns standardized: {len(categorical_cols)}")
print("Sample dtypes (first 10):")
print(X_native.dtypes.head(10))


# In[4]:


# ===========================================================================
# Feature Subset Generation
# ===========================================================================
# Each base model config is quadrupled into 4 variants, each trained on a
# different random ~50% of features.  Every feature appears in exactly 2 of
# the 4 subsets (200% total coverage).
# ===========================================================================

FEATURE_SUBSET_FRACTION = 0.45
N_FEATURE_SUBSETS = 5


def generate_feature_subsets(all_cols, n_subsets=N_FEATURE_SUBSETS,
                             fraction=FEATURE_SUBSET_FRACTION, seed=42):
    """Generate ``n_subsets`` feature subsets, each with ~``fraction`` columns.
    
    Robust greedy algorithm:
      1. Calculates the exact subset size K = round(n_total * fraction).
      2. Tracks the 'usage count' of every feature.
      3. For each subset, randomly shuffles features, then selects the K features
         that have been used the *least* so far.
         
    This mathematically guarantees that appearances remain perfectly balanced.
    If 4 subsets @ 50%, every feature appears exactly 2 times. 
    If 4 subsets @ 60% (240%), every feature appears flexibly 2 or 3 times.
    """
    all_cols = list(all_cols)
    n_total = len(all_cols)
    k = int(round(n_total * fraction))
    
    if k == 0 or k > n_total:
        raise ValueError(f"Invalid fraction={fraction} resulting in k={k} features.")

    rng = np.random.default_rng(seed)
    usage_counts = {col: 0 for col in all_cols}
    subsets = []
    
    for _ in range(n_subsets):
        # Shuffle to randomize tie-breaking for sorting
        shuffled = rng.permutation(all_cols).tolist()
        # Stable sort strictly by usage count (keeps ties in random order)
        shuffled.sort(key=lambda col: usage_counts[col])
        
        subset = shuffled[:k]
        for col in subset:
            usage_counts[col] += 1
        subsets.append(subset)

    # Validate correct distribution balance mathematically
    min_expected = (n_subsets * k) // n_total
    max_expected = min_expected + (1 if (n_subsets * k) % n_total != 0 else 0)
    
    for col in all_cols:
        count = sum(1 for sub in subsets if col in sub)
        assert min_expected <= count <= max_expected, (
            f"Feature '{col}' appeared {count} times, expected between {min_expected} and {max_expected}."
        )

    return subsets


all_feature_cols = list(X.columns)


# In[5]:


# ===========================================================================
# Level 1 Model Configuration Registry
# ===========================================================================
# Each config defines one *base* model type.  The training loop (next cell)
# creates 3 variants per config (one per feature subset), giving 39 total
# Level 1 models plus the TF-IDF text model.
#
# To tune an individual variant later, edit its param_grid below.  The 3
# grids per config start identical but are independent copies.
# ===========================================================================

def _quad_grid(grid):
    """Create 4 independent copies of a param_grid for per-variant tuning."""
    return [copy.deepcopy(grid) for _ in range(N_FEATURE_SUBSETS)]


def _make_estimator(config, fs_data):
    """Instantiate the estimator (or full pipeline) for a single variant."""
    dt = config["data_type"]

    # Build the raw estimator
    if "make_base_estimator" in config:
        base_est = config["make_base_estimator"]()
    else:
        base_est = config["estimator_cls"](**config.get("estimator_kw", {}))

    # Wrap in pipeline if needed
    if dt == "native":
        return base_est
    elif dt == "pipeline_math":
        return build_math_pipeline(base_est, num_cols=fs_data["num_cols"],
                                   cat_cols=fs_data["cat_cols"],
                                   dense_output=config.get("dense_output", False),
                                   all_num_cols=numeric_cols)
    elif dt == "pipeline_tree_ohe":
        return build_tree_ohe_pipeline(base_est, num_cols=fs_data["num_cols"],
                                       cat_cols=fs_data["cat_cols"],
                                       all_num_cols=numeric_cols)
    else:
        raise ValueError(f"Unknown data_type: {dt}")


# ---- Model configs ----

LEVEL1_MODEL_CONFIGS = []

# 1) XGBoost (Standard)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "xgb_standard",
    "data_type": "native",
    "estimator_cls": XGBClassifier,
    "estimator_kw": dict(
        objective="binary:logistic", eval_metric="aucpr",
        n_estimators=700, early_stopping_rounds=30,
        random_state=42, tree_method="hist",
        enable_categorical=True, scale_pos_weight=scale_pos_weight,
        verbose=0, n_jobs=-1,
    ),
    "param_grids": _quad_grid({
        "max_depth": [4], "learning_rate": [0.01], "subsample": [0.8],
        "colsample_bytree": [0.8], "min_child_weight": [30],
        "reg_alpha": [1.0], "reg_lambda": [2.0],
        "gamma": [0.2],
    }),
})

# 2) XGBoost (Deep & Slow)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "xgb_deep_slow",
    "data_type": "native",
    "estimator_cls": XGBClassifier,
    "estimator_kw": dict(
        objective="binary:logistic", eval_metric="aucpr",
        n_estimators=500, early_stopping_rounds=30,
        random_state=42, tree_method="hist",
        enable_categorical=True, scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    ),
    "param_grids": _quad_grid({
        "max_depth": [6], "learning_rate": [0.01], "subsample": [0.7],
        "colsample_bytree": [0.7], "reg_alpha": [2.0], "reg_lambda": [3.0],
        "gamma": [0.2],
    }),
})

# 3) XGBoost (Shallow & Fast)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "xgb_shallow_fast",
    "data_type": "native",
    "estimator_cls": XGBClassifier,
    "estimator_kw": dict(
        objective="binary:logistic", eval_metric="aucpr",
        n_estimators=200, early_stopping_rounds=30,
        random_state=42, tree_method="hist",
        enable_categorical=True, scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    ),
    "param_grids": _quad_grid({
        "max_depth": [3], "learning_rate": [0.05], "subsample": [0.7],
        "colsample_bytree": [0.9], "reg_alpha": [1.0], "reg_lambda": [2.0],
        "gamma": [0.3],
    }),
})

# 4) LightGBM (Standard)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "lgbm_standard",
    "data_type": "native",
    "estimator_cls": LGBMClassifier,
    "estimator_kw": dict(
        objective="binary", metric="average_precision",
        random_state=42, n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30, verbose=-1, n_jobs=-1,
    ),
    "param_grids": _quad_grid({
        "max_depth": [6], "learning_rate": [0.05], "num_leaves": [63],
        "subsample": [0.9], "colsample_bytree": [0.9],
        "reg_alpha": [0.0], "reg_lambda": [1.0], "min_split_gain": [0.2],
    }),
})

# 5) LightGBM (Feature Subsampler)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "lgbm_feature_subsampler",
    "data_type": "native",
    "estimator_cls": LGBMClassifier,
    "estimator_kw": dict(
        objective="binary", metric="average_precision",
        random_state=42, n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30, verbose=-1, n_jobs=-1,
    ),
    "param_grids": _quad_grid({
        "max_depth": [6], "learning_rate": [0.05], "num_leaves": [32],
        "subsample": [0.7], "colsample_bytree": [0.6],
        "reg_alpha": [1.0], "reg_lambda": [2.0], "min_split_gain": [0.1],
    }),
})

# 6) LightGBM (Row Subsampler)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "lgbm_row_subsampler",
    "data_type": "native",
    "estimator_cls": LGBMClassifier,
    "estimator_kw": dict(
        objective="binary", metric="average_precision",
        random_state=42, n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30, verbose=-1, n_jobs=-1,
    ),
    "param_grids": _quad_grid({
        "max_depth": [6], "learning_rate": [0.05], "num_leaves": [63],
        "subsample": [0.6], "subsample_freq": [1],
        "colsample_bytree": [0.9], "reg_alpha": [0.0], "reg_lambda": [1.0],
        "min_split_gain": [0.1],
    }),
})

# 7) Random Forest (Deep, Gini)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "rf_deep_gini",
    "data_type": "pipeline_tree_ohe",
    "estimator_cls": RandomForestClassifier,
    "estimator_kw": dict(
        criterion="gini", random_state=42,
        class_weight="balanced", n_jobs=-1,
    ),
    "param_grids": _quad_grid({
        "model__n_estimators": [300], "model__max_depth": [40],
        "model__min_samples_split": [2], "model__min_samples_leaf": [30],
        "model__max_features": ["sqrt"],
    }),
})

# 8) Random Forest (Entropy)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "rf_entropy",
    "data_type": "pipeline_tree_ohe",
    "estimator_cls": RandomForestClassifier,
    "estimator_kw": dict(
        criterion="entropy", random_state=42,
        class_weight="balanced", n_jobs=-1,
    ),
    "param_grids": _quad_grid({
        "model__n_estimators": [300], "model__max_depth": [40],
        "model__min_samples_split": [2], "model__min_samples_leaf": [30],
        "model__max_features": ["sqrt"],
    }),
})

# 9) Logistic Regression (L1 / Lasso)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "lr_l1",
    "data_type": "pipeline_math",
    "estimator_cls": LogisticRegression,
    "estimator_kw": dict(
        penalty="l1", solver="liblinear",
        class_weight="balanced", max_iter=350,
        random_state=42, n_jobs=-1,
    ),
    "param_grids": _quad_grid({
        "model__C": [1.0], "model__penalty": ["l1"],
        "model__solver": ["liblinear"],
    }),
})

# 10) Logistic Regression (L2 / Ridge)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "lr_l2",
    "data_type": "pipeline_math",
    "estimator_cls": LogisticRegression,
    "estimator_kw": dict(
        penalty="l2", solver="liblinear",
        class_weight="balanced", max_iter=500,
        random_state=42, n_jobs=-1,
    ),
    "param_grids": _quad_grid({
        "model__C": [1.0], "model__penalty": ["l2"],
        "model__solver": ["liblinear"],
    }),
})

# 11) SVM (RBF Kernel Approximation via Nystroem)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "svm_rbf_approx",
    "data_type": "pipeline_math",
    "make_base_estimator": lambda: Pipeline([
        ("nystroem", Nystroem(kernel="rbf", gamma=0.01, n_components=1000, random_state=42)),
        ("sgd", SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42, n_jobs=-1)),
    ]),
    "param_grids": _quad_grid({
        "model__nystroem__gamma": [0.01],
        "model__nystroem__n_components": [800],
        "model__sgd__alpha": [1e-3],
    }),
})

# 12) CatBoost (Standard)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "catboost_standard",
    "data_type": "native",
    "use_catboost_cats": True,
    "estimator_cls": CatBoostClassifier,
    "estimator_kw": dict(
        loss_function="Logloss",
        eval_metric="PRAUC:use_weights=false",
        iterations=1000, early_stopping_rounds=30,
        random_seed=42, scale_pos_weight=scale_pos_weight,
        verbose=0, allow_writing_files=False,
    ),
    "param_grids": _quad_grid({
        "depth": [4], "learning_rate": [0.02],
        "l2_leaf_reg": [5.0], "min_data_in_leaf": [10],
    }),
})

# 13) CatBoost (Deep & Slow)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "catboost_deep_slow",
    "data_type": "native",
    "use_catboost_cats": True,
    "estimator_cls": CatBoostClassifier,
    "estimator_kw": dict(
        loss_function="Logloss",
        eval_metric="BalancedAccuracy",
        random_seed=42, scale_pos_weight=scale_pos_weight,
        verbose=0, allow_writing_files=False,
    ),
    "param_grids": _quad_grid({
        "depth": [7], "learning_rate": [0.01],
        "iterations": [600], "l2_leaf_reg": [8.0],
        "min_data_in_leaf": [50],
    }),
})

# 14) LightGBM (GOSS — Gradient-based One-Side Sampling)
LEVEL1_MODEL_CONFIGS.append({
    "base_name": "lgbm_goss",
    "data_type": "native",
    "estimator_cls": LGBMClassifier,
    "estimator_kw": dict(
        objective="binary", metric="average_precision",
        boosting_type="goss",
        random_state=42, n_estimators=300,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30, verbose=-1, n_jobs=-1,
    ),
    "param_grids": _quad_grid({
        "max_depth": [6], "learning_rate": [0.05], "num_leaves": [63],
        "colsample_bytree": [0.9],
        "reg_alpha": [0.5], "reg_lambda": [1.5],
    }),
})

# 15) Naive Bayes (Gaussian) — commented out due to memory (dense OHE)
# LEVEL1_MODEL_CONFIGS.append({
#     "base_name": "naive_bayes",
#     "data_type": "pipeline_math",
#     "dense_output": True,           # GaussianNB requires dense arrays
#     "estimator_cls": GaussianNB,
#     "estimator_kw": {},
#     "param_grids": _quad_grid({
#         "model__var_smoothing": [1e-9],
#     }),
# })

print(f"Total model configs: {len(LEVEL1_MODEL_CONFIGS)}")
print(f"Total Level 1 variants (×{N_FEATURE_SUBSETS} subsets): "
      f"{len(LEVEL1_MODEL_CONFIGS) * N_FEATURE_SUBSETS}")


# ===========================================================================
# ---- GLOBAL IMPUTATION (Optional) -----------------------------------------
# ===========================================================================
if GLOBAL_IMPUTATION:
    print(f"\n{'='*72}")
    print("  Pre-imputing ALL numeric columns globally (GLOBAL_IMPUTATION=True)")
    print(f"{'='*72}")

    global_math_imputer = IterativeImputer(
        estimator=XGBRegressor(
            n_estimators=120, max_depth=3, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            random_state=42, n_jobs=-1, enable_categorical=False
        ),
        max_iter=15, random_state=42,
        sample_posterior=False, imputation_order="ascending",
    )

    X.loc[:, numeric_cols] = global_math_imputer.fit_transform(X[numeric_cols].values)
    if X_test is not None:
        X_test.loc[:, numeric_cols] = global_math_imputer.transform(X_test[numeric_cols].values)
    
    # Save for inference
    joblib.dump(global_math_imputer, RUN_DIR / "models" / "global_imputer.joblib")
    gc.collect()


# In[6]:


# ===========================================================================
# Level 1: Train all feature-subset model variants
# ===========================================================================

fs_metadata_all = {}

for config_idx, config in enumerate(LEVEL1_MODEL_CONFIGS):
    base_name = config["base_name"]
    dt = config["data_type"]

    # Generate unique subsets for this base model config
    per_model_seed = 42 + (config_idx * 1000)
    fs_cols_list = generate_feature_subsets(all_feature_cols, N_FEATURE_SUBSETS, FEATURE_SUBSET_FRACTION, seed=per_model_seed)

    for fs_idx, fs_cols in enumerate(fs_cols_list):
        variant_name = f"{base_name}_fs{fs_idx + 1}"
        param_grid = config["param_grids"][fs_idx]
        
        # Save to metadata dict
        fs_metadata_all[variant_name] = fs_cols

        # Slice data for this specific variant
        fs_num = [c for c in fs_cols if c in numeric_cols]
        fs_cat = [c for c in fs_cols if c in categorical_cols]
        fs_cat_indices = [i for i, c in enumerate(fs_cols) if c in categorical_cols]
        
        fs_data = {
            "cols": fs_cols,
            "num_cols": fs_num,
            "cat_cols": fs_cat,
            "cat_feature_indices": fs_cat_indices,
            "X_native": X_native[fs_cols],
            "X_test_native": X_test_native[fs_cols] if X_test_native is not None else None,
            "X": X[fs_cols],
            "X_test": X_test[fs_cols] if X_test is not None else None,
        }

        # Build the estimator (pipeline or raw)
        estimator = _make_estimator(config, fs_data)

        # Pick the right X data
        if dt == "native":
            X_data = fs_data["X_native"]
            X_holdout = fs_data["X_test_native"]
        else:
            X_data = X
            X_holdout = X_test

        # Build fit_params
        fit_params = {}
        if config.get("use_catboost_cats"):
            fit_params["cat_features"] = fs_data["cat_feature_indices"]

        print(f"\n{'='*72}")
        print(f"  Training: {variant_name}  ({len(fs_data['cols'])} features)")
        print(f"{'='*72}")

        run_level1_model(
            model_name=variant_name,
            estimator=estimator,
            param_grid=param_grid,
            X_data=X_data,
            y_data=y,
            use_grid_search=False,
            fit_params=fit_params,
            X_holdout=X_holdout,
        )

    # Free memory between model configs
    gc.collect()

# Save final feature subset metadata mapping
with open(RUN_DIR / "feature_subsets.json", "w") as f:
    json.dump(fs_metadata_all, f, indent=2)


# In[7]:


# # 7) MLP (Standard)
# MODEL_NAME = "mlp_standard"
# USE_GRID_SEARCH = False

# estimator = build_math_pipeline(
#     MLPClassifier(
#         hidden_layer_sizes=(64, 32, 16, 8),
#         activation="relu",
#         solver="adam",
#         alpha=1e-4,
#         batch_size=256,
#         learning_rate_init=1e-3,
#         max_iter=250,
#         early_stopping=True,
#         random_state=42,
#     )
# )

# param_grid = {
#     "model__hidden_layer_sizes": [(64, 32, 16, 8)],
#     "model__activation": ["relu"],
#     "model__alpha": [1e-4],
#     "model__learning_rate_init": [1e-3],
# }

# oof_probs_mlp_standard, test_probs_mlp_standard, report_mlp_standard = run_level1_model(
#     model_name=MODEL_NAME,
#     estimator=estimator,
#     param_grid=param_grid,
#     X_data=X,
#     y_data=y,
#     use_grid_search=USE_GRID_SEARCH,
#     X_holdout=X_test,
# )


# In[8]:


# # 8) MLP (Heavy Dropout Approximation)
# # sklearn MLPClassifier has no explicit dropout; this uses stronger regularization as a practical proxy.
# MODEL_NAME = "mlp_heavy_dropout"
# USE_GRID_SEARCH = False

# estimator = build_math_pipeline(
#     MLPClassifier(
#         hidden_layer_sizes=(128, 64, 32),
#         activation="relu",
#         solver="adam",
#         alpha=1e-2,
#         batch_size=256,
#         learning_rate_init=8e-4,
#         max_iter=300,
#         early_stopping=True,
#         random_state=42,
#     )
# )

# param_grid = {
#     "model__hidden_layer_sizes": [(128, 64, 32)],
#     "model__activation": ["relu"],
#     "model__alpha": [1e-2],
#     "model__learning_rate_init": [8e-4],
# }

# oof_probs_mlp_heavy_dropout, test_probs_mlp_heavy_dropout, report_mlp_heavy_dropout = run_level1_model(
#     model_name=MODEL_NAME,
#     estimator=estimator,
#     param_grid=param_grid,
#     X_data=X,
#     y_data=y,
#     use_grid_search=USE_GRID_SEARCH,
#     X_holdout=X_test,
# )


# In[9]:


# # 9) MLP (Different Architecture / Activation)
# MODEL_NAME = "mlp_diff_arch_activation"
# USE_GRID_SEARCH = False

# estimator = build_math_pipeline(
#     MLPClassifier(
#         hidden_layer_sizes=(192, 96, 48),
#         activation="tanh",
#         solver="adam",
#         alpha=5e-4,
#         batch_size=256,
#         learning_rate_init=1e-3,
#         max_iter=300,
#         early_stopping=True,
#         random_state=42,
#     )
# )

# param_grid = {
#     "model__hidden_layer_sizes": [(192, 96, 48)],
#     "model__activation": ["tanh"],
#     "model__alpha": [5e-4],
#     "model__learning_rate_init": [1e-3],
# }

# oof_probs_mlp_diff_arch_activation, test_probs_mlp_diff_arch_activation, report_mlp_diff_arch_activation = run_level1_model(
#     model_name=MODEL_NAME,
#     estimator=estimator,
#     param_grid=param_grid,
#     X_data=X,
#     y_data=y,
#     use_grid_search=USE_GRID_SEARCH,
#     X_holdout=X_test,
# )


# In[10]:


# # 15) KNN (K-Nearest Neighbors)
# MODEL_NAME = "knn"
# USE_GRID_SEARCH = False

# estimator = build_math_pipeline(KNeighborsClassifier(n_jobs=-1))

# param_grid = {
#     "model__n_neighbors": [15],
#     "model__weights": ["distance"],
#     "model__p": [2],
# }

# oof_probs_knn, test_probs_knn, report_knn = run_level1_model(
#     model_name=MODEL_NAME,
#     estimator=estimator,
#     param_grid=param_grid,
#     X_data=X,
#     y_data=y,
#     use_grid_search=USE_GRID_SEARCH,
#     X_holdout=X_test,
# )


# In[11]:


# # 18) MLP (Balanced Bagging via imblearn)
# from imblearn.ensemble import BalancedBaggingClassifier

# MODEL_NAME = "mlp_balanced_bagging"
# USE_GRID_SEARCH = False

# estimator = build_math_pipeline(
#     BalancedBaggingClassifier(
#         estimator=MLPClassifier(
#             hidden_layer_sizes=(128, 64, 32),
#             activation="relu",
#             alpha=1e-3,
#             max_iter=300,
#             early_stopping=True,
#             random_state=42,
#         ),
#         n_estimators=30,
#         sampling_strategy="auto",
#         replacement=True,
#         random_state=42,
#         n_jobs=-1,
#     )
# )

# param_grid = {
#     "model__n_estimators": [30],
#     "model__sampling_strategy": ["auto"],
#     "model__replacement": [True],
# }

# oof_probs_mlp_balanced_bagging, test_probs_mlp_balanced_bagging, report_mlp_balanced_bagging = run_level1_model(
#     model_name=MODEL_NAME,
#     estimator=estimator,
#     param_grid=param_grid,
#     X_data=X,
#     y_data=y,
#     use_grid_search=USE_GRID_SEARCH,
#     X_holdout=X_test,
# )


# In[12]:


# ===========================================================================
# TF-IDF Text Model (REMARKS only) — Level 1 meta-learning model
# ===========================================================================
# This model trains on the raw REMARKS text and produces OOF predictions
# that are stacked alongside the 39 structured-feature models.
# ===========================================================================

text_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=2000,
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )),
    ("model", LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=500,
        random_state=42,
        solver="lbfgs",
    )),
])

print(f"\n{'='*72}")
print("  Training: text_logreg  (TF-IDF REMARKS)")
print(f"{'='*72}")

oof_probs_text, test_probs_text, report_text = run_level1_model(
    model_name="text_logreg",
    estimator=text_pipeline,
    param_grid={},
    X_data=X_text_train,
    y_data=y,
    use_grid_search=False,
    X_holdout=X_text_test,
)

# ===========================================================================
# TF-IDF Text Model (COMMENTS only)
# ===========================================================================
text_comments_pipeline = clone(text_pipeline)

print(f"\n{'='*72}")
print("  Training: text_comments_logreg  (TF-IDF COMMENTS)")
print(f"{'='*72}")

oof_probs_text_comments, test_probs_text_comments, report_text_comments = run_level1_model(
    model_name="text_comments_logreg",
    estimator=text_comments_pipeline,
    param_grid={},
    X_data=X_text_comments_train,
    y_data=y,
    use_grid_search=False,
    X_holdout=X_text_comments_test,
)


# In[13]:


# Optional sanity checks: Level 2 feature matrices from OOF probabilities.
print(f"Models collected for Level 2 train matrix: {len(level2_train_df.columns)}")
print(level2_train_df.head(5))

if level2_test_df is not None:
    print(f"Models collected for Level 2 holdout matrix: {len(level2_test_df.columns)}")
    print(level2_test_df.head())

pd.DataFrame(model_cv_bal_acc).T.sort_values("oof_balanced_accuracy", ascending=False).head(20)


# In[14]:


# ===========================================================================
# Visualization helpers — OMIT system
# ===========================================================================

# Exact Level-1 MODEL_NAME values to omit everywhere downstream.
# (Old MLP names are irrelevant now; add any _fsN variant you want to drop.)
OMIT_MODEL_NAMES = []


def _validated_omit_names(omit_names):
    """Validate and normalize the exact-name omit list."""
    if omit_names is None:
        return []
    if not isinstance(omit_names, (list, tuple, set)):
        raise TypeError("OMIT_MODEL_NAMES must be a list/tuple/set of exact model names.")

    cleaned = []
    for name in omit_names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each omitted model name must be a non-empty string.")
        cleaned.append(name.strip())

    # Preserve order while de-duplicating.
    return list(dict.fromkeys(cleaned))


def _apply_omits_to_level2_matrices(omit_names):
    """Drop omitted model columns from Level 2 matrices if present."""
    if not omit_names:
        return

    if "level2_train_df" in globals() and isinstance(level2_train_df, pd.DataFrame):
        removed_train = [m for m in omit_names if m in level2_train_df.columns]
        if removed_train:
            level2_train_df.drop(columns=removed_train, inplace=True, errors="ignore")
            print(f"Removed from level2_train_df: {removed_train}")

    if "level2_test_df" in globals() and isinstance(level2_test_df, pd.DataFrame):
        removed_test = [m for m in omit_names if m in level2_test_df.columns]
        if removed_test:
            level2_test_df.drop(columns=removed_test, inplace=True, errors="ignore")
            print(f"Removed from level2_test_df: {removed_test}")


def _ensure_run_level1_model_omit_hook():
    """Wrap run_level1_model once so future runs respect OMIT_MODEL_NAMES."""
    if "run_level1_model" not in globals():
        return
    if globals().get("_run_level1_model_omit_hooked", False):
        return

    original_fn = globals()["run_level1_model"]

    def wrapped_run_level1_model(
        model_name: str,
        estimator,
        param_grid: dict,
        X_data,
        y_data: pd.Series,
        use_grid_search: bool = False,
        fit_params: dict | None = None,
        X_holdout=None,
    ):
        oof_probs, holdout_probs, report = original_fn(
            model_name=model_name,
            estimator=estimator,
            param_grid=param_grid,
            X_data=X_data,
            y_data=y_data,
            use_grid_search=use_grid_search,
            fit_params=fit_params,
            X_holdout=X_holdout,
        )

        omit_names = _validated_omit_names(OMIT_MODEL_NAMES)
        if model_name in omit_names:
            if "level2_train_df" in globals() and isinstance(level2_train_df, pd.DataFrame):
                level2_train_df.drop(columns=[model_name], inplace=True, errors="ignore")
            if "level2_test_df" in globals() and isinstance(level2_test_df, pd.DataFrame):
                level2_test_df.drop(columns=[model_name], inplace=True, errors="ignore")
            print(f"{model_name} omitted from Level 2 matrices via OMIT_MODEL_NAMES.")

        return oof_probs, holdout_probs, report

    globals()["run_level1_model"] = wrapped_run_level1_model
    globals()["_run_level1_model_omit_hooked"] = True


def _filter_probability_df(prob_df: pd.DataFrame, omit_names):
    """Return probability frame with omitted model columns removed."""
    if not omit_names:
        return prob_df

    keep_cols = [c for c in prob_df.columns if c not in omit_names]
    omitted_present = [c for c in omit_names if c in prob_df.columns]

    if omitted_present:
        print(f"Omitting from visualization: {omitted_present}")

    if not keep_cols:
        raise ValueError("All model columns were omitted. Add at least one model back.")

    return prob_df.loc[:, keep_cols]


def build_correctness_from_probabilities(prob_df: pd.DataFrame, y_true: pd.Series, threshold: float = 0.5):
    """Convert model probability outputs into hard predictions and correctness masks."""
    if not isinstance(prob_df, pd.DataFrame):
        raise TypeError("prob_df must be a pandas DataFrame.")

    y_aligned = pd.Series(y_true).reindex(prob_df.index)
    if y_aligned.isna().any():
        raise ValueError("y_true could not be fully aligned to prob_df index.")

    y_aligned = y_aligned.astype(int)
    pred_df = (prob_df >= threshold).astype(int)
    correctness_df = pred_df.eq(y_aligned, axis=0)

    return pred_df, correctness_df, y_aligned


def _sample_indices_by_damage_ratio(y_aligned: pd.Series, k: int, damage_fraction: float, seed: int = 42):
    """Sample indices so positives are close to the requested fraction."""
    if not (0.0 <= damage_fraction <= 1.0):
        raise ValueError("damage_fraction must be between 0.0 and 1.0.")

    sample_n = min(k, len(y_aligned))
    if sample_n <= 0:
        raise ValueError("k must be >= 1.")

    pos_idx = y_aligned[y_aligned == 1].index
    neg_idx = y_aligned[y_aligned == 0].index

    n_pos_target = int(round(sample_n * damage_fraction))
    n_neg_target = sample_n - n_pos_target

    n_pos = min(n_pos_target, len(pos_idx))
    n_neg = min(n_neg_target, len(neg_idx))

    # Reallocate any unmet target to whichever class still has available rows.
    remaining = sample_n - (n_pos + n_neg)
    if remaining > 0:
        pos_spare = max(0, len(pos_idx) - n_pos)
        add_pos = min(remaining, pos_spare)
        n_pos += add_pos
        remaining -= add_pos

    if remaining > 0:
        neg_spare = max(0, len(neg_idx) - n_neg)
        add_neg = min(remaining, neg_spare)
        n_neg += add_neg
        remaining -= add_neg

    rng = np.random.default_rng(seed)

    sampled_pos = rng.choice(np.array(pos_idx), size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=object)
    sampled_neg = rng.choice(np.array(neg_idx), size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=object)

    sampled_idx = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(sampled_idx)

    return pd.Index(sampled_idx)


def _entropy_sorted_matrix(correctness_sub_df: pd.DataFrame) -> pd.DataFrame:
    """Sort correctness matrix rows by disagreement entropy p(1-p)."""
    if correctness_sub_df.empty:
        return correctness_sub_df

    sample_int = correctness_sub_df.astype(int)
    row_p = sample_int.mean(axis=1)
    sample_int["Entropy"] = row_p * (1 - row_p)
    sample_int = sample_int.sort_values(by="Entropy", ascending=False)
    sample_int = sample_int.drop(columns=["Entropy"])
    return sample_int


def _plot_correctness_heatmap(sample_int: pd.DataFrame, title: str):
    """Render one correctness heatmap panel."""
    if sample_int.empty:
        print(f"{title}: no rows to display.")
        return

    fig_height = max(8, len(sample_int) // 40)
    plt.figure(figsize=(max(10, len(sample_int.columns) * 0.8), fig_height))

    cmap = sns.color_palette(["#ff4d4d", "#4dff4d"]) if sns else ["red", "green"]

    if sns:
        sns.heatmap(
            sample_int,
            cmap=cmap,
            cbar=False,
            yticklabels=False,
            linewidths=0.0 if len(sample_int) > 500 else 0.35,
        )
    else:
        plt.imshow(sample_int.values, aspect="auto", cmap="RdYlGn")

    plt.title(title)
    plt.xlabel("Base Models")
    plt.ylabel("Records (Most Debated at Top -> Unanimous at Bottom)")
    plt.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.xticks(rotation=45, ha="left")
    plt.tight_layout()
    _savefig(plt, title.replace(" ", "_").replace("-", "_")[:60])


def plot_entropy_sample(
    correctness_df: pd.DataFrame,
    y_true: pd.Series,
    k: int = 1000,
    damage_fraction: float = 0.8,
    seed: int = 42,
    split_by_actual_label: bool = True,
):
    """Plot correctness heatmaps from sampled rows, optionally split by true label."""
    if correctness_df.empty:
        raise ValueError("correctness_df is empty.")

    y_aligned = pd.Series(y_true).reindex(correctness_df.index).astype(int)
    sample_idx = _sample_indices_by_damage_ratio(y_aligned, k=k, damage_fraction=damage_fraction, seed=seed)
    sample_df = correctness_df.loc[sample_idx].copy()

    actual_damage_ratio = float(y_aligned.loc[sample_idx].mean()) if len(sample_idx) > 0 else float("nan")
    sample_damage_mask = y_aligned.loc[sample_idx] == 1
    n_damage = int(sample_damage_mask.sum())
    n_no_damage = int((~sample_damage_mask).sum())

    print(
        f"Requested damage fraction={damage_fraction:.2f}; "
        f"actual sampled damage fraction={actual_damage_ratio:.2f}; sample size={len(sample_idx)}"
    )
    print(f"Sample split -> damage rows: {n_damage}, no-damage rows: {n_no_damage}")

    if split_by_actual_label:
        damage_df = _entropy_sorted_matrix(sample_df.loc[sample_damage_mask])
        no_damage_df = _entropy_sorted_matrix(sample_df.loc[~sample_damage_mask])

        _plot_correctness_heatmap(
            damage_df,
            "Model Correctness Grid - Actual DAMAGE Rows",
        )
        _plot_correctness_heatmap(
            no_damage_df,
            "Model Correctness Grid - Actual NO DAMAGE Rows",
        )
    else:
        combined_df = _entropy_sorted_matrix(sample_df)
        _plot_correctness_heatmap(
            combined_df,
            "Model Correctness Grid (Combined Sample, Sorted by Entropy)",
        )


def plot_single_record_correctness(
    prob_df: pd.DataFrame,
    y_true: pd.Series,
    record_idx,
    threshold: float = 0.5,
):
    """Show per-model correctness (red/green) and probabilities for one record."""
    pred_df, correctness_df, y_aligned = build_correctness_from_probabilities(prob_df, y_true, threshold)

    if record_idx not in prob_df.index:
        raise KeyError(f"record_idx={record_idx} not found in prob_df index.")

    row_prob = prob_df.loc[record_idx]
    row_pred = pred_df.loc[record_idx]
    row_correct = correctness_df.loc[record_idx]
    y_value = int(y_aligned.loc[record_idx])

    single_row = pd.DataFrame([row_correct.astype(int).values], columns=prob_df.columns, index=[str(record_idx)])

    plt.figure(figsize=(max(10, len(prob_df.columns) * 0.8), 2.6))
    if sns:
        cmap = sns.color_palette(["#ff4d4d", "#4dff4d"])
        sns.heatmap(single_row, cmap=cmap, cbar=False, yticklabels=True, linewidths=0.4)
    plt.title(f"Record {record_idx} Correctness by Model (True label={y_value})")
    plt.xlabel("Base Models")
    plt.ylabel("Record")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _savefig(plt, f"record_{record_idx}_correctness")

    detail_df = pd.DataFrame(
        {
            "prob_damage": row_prob,
            "pred_label": row_pred,
            "is_correct": row_correct,
        }
    )

    plt.figure(figsize=(max(10, len(detail_df) * 0.55), 4.2))
    colors = ["#4dff4d" if ok else "#ff4d4d" for ok in detail_df["is_correct"].tolist()]
    plt.bar(detail_df.index, detail_df["prob_damage"], color=colors)
    plt.axhline(threshold, linestyle="--", linewidth=1, color="black", label=f"threshold={threshold}")
    plt.ylim(0, 1)
    plt.title(f"Record {record_idx}: Probability of Damage by Model")
    plt.ylabel("Predicted probability (class 1)")
    plt.xlabel("Base Models")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    _savefig(plt, f"record_{record_idx}_probabilities")

    return detail_df


def visualize_level2_predictions(
    prob_df: pd.DataFrame,
    y_true: pd.Series,
    k: int = 1000,
    damage_fraction: float = 0.8,
    seed: int = 42,
    threshold: float = 0.5,
    record_idx=None,
    split_by_actual_label: bool = True,
):
    """Convenience wrapper for full-grid and optional single-record plots."""
    omit_names = _validated_omit_names(OMIT_MODEL_NAMES)

    # Keep Level 2 matrices consistent with the same omission list.
    _apply_omits_to_level2_matrices(omit_names)

    # Ensure visualizations also honor the same omission list.
    prob_df_filtered = _filter_probability_df(prob_df, omit_names)

    _, correctness_df, y_aligned = build_correctness_from_probabilities(prob_df_filtered, y_true, threshold)

    print(f"Total records: {len(correctness_df):,}")
    print(f"Total base models (columns): {correctness_df.shape[1]}")

    plot_entropy_sample(
        correctness_df,
        y_true=y_aligned,
        k=k,
        damage_fraction=damage_fraction,
        seed=seed,
        split_by_actual_label=split_by_actual_label,
    )

    if record_idx is not None:
        detail_df = plot_single_record_correctness(prob_df_filtered, y_aligned, record_idx=record_idx, threshold=threshold)
        return correctness_df, detail_df

    return correctness_df


# Install the run_level1_model hook once, then apply current omit list immediately.
_ensure_run_level1_model_omit_hook()
_apply_omits_to_level2_matrices(_validated_omit_names(OMIT_MODEL_NAMES))


# Example usage:
# OMIT_MODEL_NAMES = ["catboost_deep_slow_fs1"]
# correctness_df = visualize_level2_predictions(level2_train_df, y, k=25, damage_fraction=0.8, seed=42)
correctness_df, record_detail = visualize_level2_predictions(
    level2_train_df,
    y,
    k=7000,
    damage_fraction=0.8,
    seed=0,
    record_idx=level2_train_df.index[0],
    split_by_actual_label=True,
)


# In[ ]:





# In[15]:


# =========================
# Level 2 (Meta-Model) Setup
# =========================

if level2_train_df.empty:
    raise ValueError("level2_train_df is empty. Run Level 1 base models first.")

if len(level2_train_df) != len(y):
    raise ValueError("level2_train_df and y are not aligned by row count.")

# ===========================================================================
# ---- INJECT RAW FEATURES INTO LEVEL 2 MATRIX ------------------------------
# ===========================================================================
L2_RAW_FEATURES = ["AIRCRAFT", "SIZE", "PERSON"]
for col in L2_RAW_FEATURES:
    # Cast to string to ensure safe encoding (avoids mixed-type or category issues)
    level2_train_df[col] = X[col].astype(str)
    if level2_test_df is not None:
        level2_test_df[col] = X_test[col].astype(str)

# Save a global Target Encoder for inference.py to use on the test set
l2_te_global = TargetEncoder(random_state=42)
l2_te_global.fit(level2_train_df[L2_RAW_FEATURES], y)
if RUN_DIR is not None:
    joblib.dump(l2_te_global, RUN_DIR / "models" / "l2_target_encoder.joblib")

l2_model_oof_predictions = {}
l2_model_test_predictions = {}
l2_model_cv_bal_acc = {}

# If you train multiple Level 2 models, these can be used as Level 3 inputs.
level3_train_df = pd.DataFrame(index=level2_train_df.index)
level3_test_df = pd.DataFrame(index=level2_test_df.index) if level2_test_df is not None else None


def find_best_balanced_threshold(y_true: pd.Series, probs: np.ndarray, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.95, 181)

    best_thr = 0.5
    best_score = -1.0

    y_true_arr = np.asarray(y_true).astype(int)
    probs_arr = np.asarray(probs, dtype=float)

    for thr in grid:
        pred = (probs_arr >= thr).astype(int)
        score = balanced_accuracy_score(y_true_arr, pred)
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return best_thr, float(best_score)


def run_level2_model(
    model_name: str,
    estimator,
    param_grid: dict,
    X_data: pd.DataFrame,
    y_data: pd.Series,
    use_grid_search: bool = False,
    fit_params: dict | None = None,
    X_holdout: pd.DataFrame | None = None,
):
    fit_params = fit_params or {}
    oof_probs = np.zeros(len(X_data), dtype=float)
    holdout_probs = np.zeros(len(X_holdout), dtype=float) if X_holdout is not None else None
    fold_bal_acc = []
    chosen_params = first_grid_point(param_grid)
    overall_pos_rate = float(y_data.mean())
    strat_tol = max(0.002, 1.0 / max(len(y_data) // skf.n_splits, 1))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_data, y_data), start=1):
        X_tr, X_val = X_data.iloc[tr_idx].copy(), X_data.iloc[val_idx].copy()
        y_tr, y_val = y_data.iloc[tr_idx], y_data.iloc[val_idx]
        
        X_holdout_cv = None
        if X_holdout is not None:
            X_holdout_cv = X_holdout.copy()

        # Target Encode the raw features specifically for Level 2 models
        if isinstance(X_tr, pd.DataFrame):
            enc_cols = [c for c in ["AIRCRAFT", "SIZE", "PERSON"] if c in X_tr.columns]
            if enc_cols:
                te_cv = TargetEncoder(random_state=42)
                X_tr[enc_cols] = te_cv.fit_transform(X_tr[enc_cols].astype(str), y_tr)
                X_val[enc_cols] = te_cv.transform(X_val[enc_cols].astype(str))
                for col in enc_cols:
                    X_tr[col] = X_tr[col].astype("float64")
                    X_val[col] = X_val[col].astype("float64")

                if X_holdout_cv is not None and all(c in X_holdout_cv.columns for c in enc_cols):
                    X_holdout_cv[enc_cols] = te_cv.transform(X_holdout_cv[enc_cols].astype(str))
                    for col in enc_cols:
                        X_holdout_cv[col] = X_holdout_cv[col].astype("float64")

        tr_pos_rate = float(y_tr.mean())
        val_pos_rate = float(y_val.mean())
        tr_neg_rate = 1.0 - tr_pos_rate
        val_neg_rate = 1.0 - val_pos_rate

        print(
            f"{model_name} | Fold {fold} stratification -> "
            f"train no/yes={tr_neg_rate:.4f}/{tr_pos_rate:.4f}, "
            f"val no/yes={val_neg_rate:.4f}/{val_pos_rate:.4f}, "
            f"global no/yes={(1.0 - overall_pos_rate):.4f}/{overall_pos_rate:.4f}"
        )

        if abs(tr_pos_rate - overall_pos_rate) > strat_tol or abs(val_pos_rate - overall_pos_rate) > strat_tol:
            raise ValueError(
                f"Stratification drift in fold {fold}. "
                f"train_pos={tr_pos_rate:.6f}, val_pos={val_pos_rate:.6f}, "
                f"global_pos={overall_pos_rate:.6f}, tol={strat_tol:.6f}"
            )

        fold_model = clone(estimator)

        if use_grid_search:
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid = GridSearchCV(
                estimator=fold_model,
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_tr, y_tr, **fit_params)
            fitted_model = grid.best_estimator_
            chosen_params = grid.best_params_
        else:
            fitted_model = fold_model.set_params(**chosen_params)
            fitted_model.fit(X_tr, y_tr, **fit_params)

        # Save Level 2 fold model
        if SAVE_FOLD_MODELS and RUN_DIR is not None:
            model_path = RUN_DIR / "models" / f"{model_name}_fold{fold}.joblib"
            try:
                joblib.dump(fitted_model, model_path)
            except Exception as e:
                print(f"  Warning: could not save L2 model {model_name} fold {fold}: {e}")

        if hasattr(fitted_model, "predict_proba"):
            val_probs = fitted_model.predict_proba(X_val)[:, 1]
        else:
            val_scores = fitted_model.decision_function(X_val)
            val_probs = 1.0 / (1.0 + np.exp(-val_scores))

        oof_probs[val_idx] = val_probs

        val_pred = (val_probs >= 0.5).astype(int)
        fold_score = balanced_accuracy_score(y_val, val_pred)
        fold_bal_acc.append(float(fold_score))

        if X_holdout is not None:
            eval_holdout = X_holdout_cv if X_holdout_cv is not None else X_holdout
            if hasattr(fitted_model, "predict_proba"):
                fold_holdout_probs = fitted_model.predict_proba(eval_holdout)[:, 1]
            else:
                holdout_scores = fitted_model.decision_function(eval_holdout)
                fold_holdout_probs = 1.0 / (1.0 + np.exp(-holdout_scores))
            holdout_probs += fold_holdout_probs / skf.n_splits

        print(f"{model_name} | Fold {fold} balanced_accuracy: {fold_score:.4f}")

    oof_bal_acc_05 = balanced_accuracy_score(y_data, (oof_probs >= 0.5).astype(int))
    best_thr, oof_bal_acc_best = find_best_balanced_threshold(y_data, oof_probs)

    l2_model_oof_predictions[model_name] = oof_probs
    l2_model_cv_bal_acc[model_name] = {
        "fold_balanced_accuracy": fold_bal_acc,
        "mean_balanced_accuracy": float(np.mean(fold_bal_acc)),
        "oof_balanced_accuracy_0.5": float(oof_bal_acc_05),
        "oof_balanced_accuracy_best": float(oof_bal_acc_best),
        "best_threshold": float(best_thr),
        "chosen_params": {k: str(v) for k, v in chosen_params.items()},
    }

    if holdout_probs is not None:
        l2_model_test_predictions[model_name] = holdout_probs

    level3_train_df[model_name] = oof_probs
    if level3_test_df is not None and holdout_probs is not None:
        level3_test_df[model_name] = holdout_probs

    print(f"{model_name} | OOF balanced_accuracy @0.5: {oof_bal_acc_05:.4f}")
    print(f"{model_name} | OOF balanced_accuracy @best_threshold={best_thr:.3f}: {oof_bal_acc_best:.4f}")

    return oof_probs, holdout_probs, l2_model_cv_bal_acc[model_name]


print(f"Level 2 train matrix shape: {level2_train_df.shape}")
print(f"Level 2 test matrix shape: {None if level2_test_df is None else level2_test_df.shape}")
level2_train_df.head(10)


# In[16]:


# Level 2 - Baseline: Logistic Regression
MODEL_NAME = "l2_logistic_baseline"
USE_GRID_SEARCH = False

# A simple, strictly regularized linear model is the gold standard for stacking
estimator = LogisticRegression(
    penalty="l2",
    C=0.3, # Strong regularization to prevent trusting any one model too much
    solver="lbfgs",
    max_iter=500,
    random_state=42,
    class_weight={0: 1.0, 1: 13.0}
)

# No param grid needed for the baseline
param_grid = {"C": [0.3]}

oof_probs_l2_lr, test_probs_l2_lr, report_l2_lr = run_level2_model(
    model_name=MODEL_NAME,
    estimator=estimator,
    param_grid=param_grid,
    X_data=level2_train_df,
    y_data=y,
    use_grid_search=USE_GRID_SEARCH,
    X_holdout=level2_test_df,
)


# In[17]:


# Level 2 - Model 1: XGBoost (Standard Defaults) — FIXED
MODEL_NAME = "l2_xgb_standard"
USE_GRID_SEARCH = False

# Correct way for XGBoost: use scale_pos_weight (NOT class_weight)
estimator = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=150,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
    
    # === THIS IS THE KEY CHANGE ===
    scale_pos_weight=8.0,          # mild weighting (matches your logistic's 4.0–6.0)
                                   # Full imbalance was ~15.7 → we use ~40% of that
                                   # → threshold moves from ~0.05 → ~0.22–0.32
)

param_grid = {
    "max_depth": [2, 3],
    "learning_rate": [0.05],
    "subsample": [1.0],
    "colsample_bytree": [1.0],
    "reg_alpha": [1.0],
    "reg_lambda": [10.0],
    # Optional: you can grid the weight later if you want
    "scale_pos_weight": [13],
}

oof_probs_l2_xgb_standard, test_probs_l2_xgb_standard, report_l2_xgb_standard = run_level2_model(
    model_name=MODEL_NAME,
    estimator=estimator,
    param_grid=param_grid,
    X_data=level2_train_df,
    y_data=y,
    use_grid_search=USE_GRID_SEARCH,
    X_holdout=level2_test_df,
)

pd.DataFrame(l2_model_cv_bal_acc).T.sort_values("oof_balanced_accuracy_best", ascending=False)



# In[19]:


# === DIAGNOSTIC 1: Diversity & Correlation ===

corr = level2_train_df.select_dtypes(include=[np.number]).corr(method='pearson').abs()
avg_corr = corr.mean().mean()
print(f"Average pairwise correlation among Level 1 models: {avg_corr:.3f} "
      f"(ideal < 0.75 for good stacking)")

plt.figure(figsize=(20, 16))
if sns:
    sns.heatmap(corr, annot=False, cmap="coolwarm", fmt=".2f", vmin=0.6, vmax=1.0)
plt.title("Level 1 OOF Probability Correlations — High = Bad Diversity")
_savefig(plt, "level1_correlation_heatmap")

# Quick prune list
high_corr = (corr > 0.88).sum().sort_values(ascending=False)
print("Models correlated with >2 others (>0.88):")
print(high_corr[high_corr > 2])


# In[20]:


# === DIAGNOSTIC 2: Score Breakdown ===
l1_scores = pd.DataFrame(model_cv_bal_acc).T[['oof_balanced_accuracy']].round(4)
l1_scores = l1_scores.rename(columns={'oof_balanced_accuracy': 'Level1_@0.5'})

l2_scores = pd.DataFrame(l2_model_cv_bal_acc).T[['oof_balanced_accuracy_best', 'best_threshold']].round(4)

print("Top 10 Level 1 models:")
print(l1_scores.sort_values('Level1_@0.5', ascending=False).head(10))

print("\nCurrent Level 2 / Stack scores:")
print(l2_scores)

print(f"\nBest single Level 1: {l1_scores['Level1_@0.5'].max():.4f}")
print(f"Current best stack: {l2_scores['oof_balanced_accuracy_best'].max():.4f}")
print(f"Stacking gain so far: +{l2_scores['oof_balanced_accuracy_best'].max() - l1_scores['Level1_@0.5'].max():.4f}")


# In[21]:


# === DIAGNOSTIC 3: Inspect Hard Cases From The Grid Sample ===

# Match these to the settings used in your grid visualization cell.
GRID_SAMPLE_K = 7000
GRID_DAMAGE_FRACTION = 0.8
GRID_SEED = 0
GRID_THRESHOLD = 0.5

# Filtering controls:
TARGET_CLASS = 1            # 1 = damage, 0 = no damage, None = both
ONLY_ALL_MODELS_WRONG = True
ENTROPY_MIN = None          # e.g. 0.15
ENTROPY_MAX = None          # e.g. 0.25 (entropy range is always in [0, 0.25])
MAX_ROWS_TO_DISPLAY = 600
HEATMAP_MAX_ROWS = 2000

# For a row-level vote summary (independent of l2_vote_60pct_rule training):
BASE_MODEL_PROB_THRESHOLD = 0.5
ROW_VOTE_THRESHOLD = 0.6

omit_names = _validated_omit_names(OMIT_MODEL_NAMES)
prob_df_grid = _filter_probability_df(level2_train_df.select_dtypes(include=[np.number]), omit_names)

pred_df_grid, correctness_df_grid, y_aligned_grid = build_correctness_from_probabilities(
    prob_df_grid,
    y,
    threshold=GRID_THRESHOLD,
)

sample_idx = _sample_indices_by_damage_ratio(
    y_aligned_grid,
    k=GRID_SAMPLE_K,
    damage_fraction=GRID_DAMAGE_FRACTION,
    seed=GRID_SEED,
)

sample_prob_df = prob_df_grid.loc[sample_idx]
sample_correctness_df = correctness_df_grid.loc[sample_idx]
sample_y = y_aligned_grid.loc[sample_idx]

n_models = sample_correctness_df.shape[1]
row_correct_count = sample_correctness_df.sum(axis=1).astype(int)
row_correct_rate = row_correct_count / max(n_models, 1)
row_entropy = row_correct_rate * (1.0 - row_correct_rate)
row_all_wrong = row_correct_count.eq(0)
row_vote_share = (sample_prob_df >= BASE_MODEL_PROB_THRESHOLD).mean(axis=1)
row_vote_pred = (row_vote_share >= ROW_VOTE_THRESHOLD).astype(int)

meta_df = pd.DataFrame(
    {
        "true_label": sample_y,
        "row_correct_count": row_correct_count,
        "row_incorrect_count": n_models - row_correct_count,
        "row_correct_rate": row_correct_rate,
        "row_entropy": row_entropy,
        "all_models_wrong": row_all_wrong,
        "row_vote_share_damage": row_vote_share,
        "row_vote_pred": row_vote_pred,
    },
    index=sample_idx,
)

mask = pd.Series(True, index=meta_df.index)
if TARGET_CLASS in (0, 1):
    mask &= meta_df["true_label"].eq(TARGET_CLASS)
if ONLY_ALL_MODELS_WRONG:
    mask &= meta_df["all_models_wrong"]
if ENTROPY_MIN is not None:
    mask &= meta_df["row_entropy"] >= float(ENTROPY_MIN)
if ENTROPY_MAX is not None:
    mask &= meta_df["row_entropy"] <= float(ENTROPY_MAX)

selected_idx = meta_df.index[mask]
selected_meta = meta_df.loc[selected_idx].sort_values("row_entropy", ascending=False)
selected_features = X.loc[selected_idx]

print(f"Grid sample size: {len(sample_idx):,} | Models used: {n_models}")
print(f"Selected rows after filters: {len(selected_idx):,}")
if TARGET_CLASS in (0, 1):
    print(f"Target class filter: {TARGET_CLASS}")
print(f"All-models-wrong filter: {ONLY_ALL_MODELS_WRONG}")
print(f"Entropy range filter: [{ENTROPY_MIN}, {ENTROPY_MAX}]")

if len(selected_idx) == 0:
    print("No rows matched your current filters. Relax class/all-wrong/entropy settings.")
else:
    print("\nSelected rows (summary):")
    display(selected_meta.head(MAX_ROWS_TO_DISPLAY))

    print("\nSelected rows with original feature columns:")
    display(pd.concat([selected_meta, selected_features], axis=1).head(MAX_ROWS_TO_DISPLAY))

    # 1) Entropy distribution in the sampled grid, with selected subset overlay.
    plt.figure(figsize=(10, 4.5))
    if sns:
        sns.histplot(meta_df["row_entropy"], bins=25, color="#4c78a8", alpha=0.50, label="all sampled rows")
        sns.histplot(selected_meta["row_entropy"], bins=25, color="#f58518", alpha=0.65, label="selected rows")
    if ENTROPY_MIN is not None:
        plt.axvline(float(ENTROPY_MIN), color="black", linestyle="--", linewidth=1, label="entropy min")
    if ENTROPY_MAX is not None:
        plt.axvline(float(ENTROPY_MAX), color="gray", linestyle="--", linewidth=1, label="entropy max")
    plt.title("Row Entropy Distribution (Sampled Grid)")
    plt.xlabel("Entropy = p(1-p), where p = per-row model correctness rate")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    _savefig(plt, "hard_cases_entropy_dist")

    # 2) Error rate by model on the selected rows.
    selected_correctness = sample_correctness_df.loc[selected_idx]
    model_error_rate = (1.0 - selected_correctness.mean(axis=0)).sort_values(ascending=False)

    plt.figure(figsize=(max(11, 0.65 * len(model_error_rate)), 4.8))
    if sns:
        sns.barplot(x=model_error_rate.index, y=model_error_rate.values, color="#e45756")
    plt.ylim(0, 1)
    plt.title("Model Error Rate On Selected Rows")
    plt.xlabel("Base Models")
    plt.ylabel("Error rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _savefig(plt, "hard_cases_model_error_rate")

    # 3) Correctness heatmap for selected rows (high entropy first).
    heatmap_df = selected_correctness.astype(int)
    heatmap_df["_entropy"] = selected_meta["row_entropy"]
    heatmap_df = heatmap_df.sort_values("_entropy", ascending=False).drop(columns=["_entropy"])
    heatmap_df = heatmap_df.head(HEATMAP_MAX_ROWS)

    plt.figure(figsize=(max(10, 0.75 * heatmap_df.shape[1]), max(4.5, heatmap_df.shape[0] / 18)))
    if sns:
        sns.heatmap(
            heatmap_df,
            cmap=sns.color_palette(["#ff4d4d", "#4dff4d"]),
            cbar=False,
            yticklabels=False,
            linewidths=0.0 if len(heatmap_df) > 300 else 0.30,
        )
    plt.title("Selected Rows Correctness Heatmap (Red=Wrong, Green=Correct)")
    plt.xlabel("Base Models")
    plt.ylabel("Selected rows (entropy-sorted)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _savefig(plt, "hard_cases_correctness_heatmap")

    # 4) Optional numeric feature shift: selected rows vs class baseline in sampled grid.
    numeric_cols_in_X = X.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols_in_X:
        if TARGET_CLASS in (0, 1):
            baseline_idx = meta_df.index[meta_df["true_label"].eq(TARGET_CLASS)]
        else:
            baseline_idx = meta_df.index

        baseline_num = X.loc[baseline_idx, numeric_cols_in_X]
        selected_num = X.loc[selected_idx, numeric_cols_in_X]

        baseline_mean = baseline_num.mean()
        selected_mean = selected_num.mean()
        baseline_std = baseline_num.std(ddof=0).replace(0, np.nan)
        effect = ((selected_mean - baseline_mean).abs() / baseline_std).replace([np.inf, -np.inf], np.nan).dropna()

        top_shift = effect.sort_values(ascending=False).head(12)
        if len(top_shift) > 0:
            plt.figure(figsize=(8, 5.2))
            if sns:
                sns.barplot(x=top_shift.values, y=top_shift.index, color="#72b7b2")
            plt.title("Top Numeric Feature Shifts (Selected vs Baseline)")
            plt.xlabel("|mean_selected - mean_baseline| / std_baseline")
            plt.ylabel("Feature")
            plt.tight_layout()
            _savefig(plt, "hard_cases_feature_shifts")


# In[22]:


# ===========================================================================
# Level 3: Weighted Average Blender (50/50)
# ===========================================================================

L3_MODEL_NAMES = ["l2_logistic_baseline", "l2_xgb_standard"]
L3_WEIGHTS = {name: 0.5 for name in L3_MODEL_NAMES}

print(f"\n{'='*72}")
print("  Level 3: Weighted Average Blender")
print(f"{'='*72}")
print(f"Input models: {L3_MODEL_NAMES}")
print(f"Weights: {L3_WEIGHTS}")

# Verify all Level 2 models are present in level3_train_df
for name in L3_MODEL_NAMES:
    if name not in level3_train_df.columns:
        raise ValueError(f"Level 2 model '{name}' not found in level3_train_df. "
                         f"Available: {list(level3_train_df.columns)}")

# --- OOF blend ---
l3_oof_probs = sum(level3_train_df[name] * w for name, w in L3_WEIGHTS.items())

# --- Test blend ---
l3_test_probs = None
if level3_test_df is not None:
    l3_test_probs = sum(level3_test_df[name] * w for name, w in L3_WEIGHTS.items())

# --- Threshold sweep for balanced accuracy ---
best_l3_thr, best_l3_bal_acc = find_best_balanced_threshold(y, l3_oof_probs)

# Balanced accuracy at default 0.5 threshold
l3_bal_acc_05 = balanced_accuracy_score(y, (l3_oof_probs >= 0.5).astype(int))

# AUCPR (average_precision_score)
l3_avg_precision = average_precision_score(y, l3_oof_probs)

# Store Level 3 metrics
l3_metrics = {
    "oof_balanced_accuracy_0.5": float(l3_bal_acc_05),
    "oof_balanced_accuracy_best": float(best_l3_bal_acc),
    "best_threshold": float(best_l3_thr),
    "oof_average_precision": float(l3_avg_precision),
    "weights": {k: float(v) for k, v in L3_WEIGHTS.items()},
    "input_models": L3_MODEL_NAMES,
}

print(f"\nLevel 3 | OOF balanced_accuracy @0.5: {l3_bal_acc_05:.4f}")
print(f"Level 3 | OOF balanced_accuracy @best_threshold={best_l3_thr:.3f}: {best_l3_bal_acc:.4f}")
print(f"Level 3 | OOF average_precision (AUCPR): {l3_avg_precision:.4f}")

# --- Per-L2-model contribution bar chart ---
plt.figure(figsize=(8, 4))
model_names_short = [n.replace("l2_", "") for n in L3_MODEL_NAMES]
weights_vals = [L3_WEIGHTS[n] for n in L3_MODEL_NAMES]
bars = plt.bar(model_names_short, weights_vals, color=["#4c78a8", "#f58518", "#54a24b"])
for bar, w in zip(bars, weights_vals):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f"{w:.3f}", ha="center", va="bottom", fontsize=11)
plt.title("Level 3 Blender — Model Weights")
plt.ylabel("Weight")
plt.ylim(0, 0.5)
plt.tight_layout()
_savefig(plt, "l3_model_weights")

# --- Level 3 threshold sweep visualization ---
thresholds = np.linspace(0.01, 0.95, 181)
bal_accs = [balanced_accuracy_score(y, (l3_oof_probs >= t).astype(int)) for t in thresholds]

plt.figure(figsize=(10, 4.5))
plt.plot(thresholds, bal_accs, color="#4c78a8", linewidth=2)
plt.axvline(best_l3_thr, color="#e45756", linestyle="--", linewidth=1.5,
            label=f"Best threshold = {best_l3_thr:.3f}")
plt.axhline(best_l3_bal_acc, color="#72b7b2", linestyle=":", linewidth=1,
            label=f"Best bal_acc = {best_l3_bal_acc:.4f}")
plt.title("Level 3 — Balanced Accuracy vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Balanced Accuracy")
plt.legend()
plt.tight_layout()
_savefig(plt, "l3_threshold_sweep")

# --- Display all Level 2 & Level 3 together ---
print("\n=== Level 2 Individual Model Scores ===")
for name in L3_MODEL_NAMES:
    report = l2_model_cv_bal_acc[name]
    l2_ap = average_precision_score(y, l2_model_oof_predictions[name])
    print(f"  {name}:")
    print(f"    Balanced Accuracy @best_thr={report['best_threshold']:.3f}: "
          f"{report['oof_balanced_accuracy_best']:.4f}")
    print(f"    Average Precision (AUCPR): {l2_ap:.4f}")

print(f"\n=== Level 3 Final Score ===")
print(f"  Balanced Accuracy @best_thr={best_l3_thr:.3f}: {best_l3_bal_acc:.4f}")
print(f"  Average Precision (AUCPR): {l3_avg_precision:.4f}")


# In[23]:


# ===========================================================================
# Save run metadata — all metrics, OOF predictions, and test predictions
# ===========================================================================

# 1) Level 1 metrics
with open(RUN_DIR / "level1_metrics.json", "w") as f:
    json.dump(model_cv_bal_acc, f, indent=2, default=str)

# 2) Level 2 metrics
with open(RUN_DIR / "level2_metrics.json", "w") as f:
    json.dump(l2_model_cv_bal_acc, f, indent=2, default=str)

# 3) Level 3 metrics
with open(RUN_DIR / "level3_metrics.json", "w") as f:
    json.dump(l3_metrics, f, indent=2, default=str)

# 4) Level 3 weights & threshold (for inference)
with open(RUN_DIR / "models" / "l3_weights.json", "w") as f:
    json.dump(L3_WEIGHTS, f, indent=2)

with open(RUN_DIR / "models" / "l3_best_threshold.json", "w") as f:
    json.dump({"best_threshold": float(best_l3_thr)}, f, indent=2)

# 5) Run metadata
run_meta = {
    "timestamp": datetime.datetime.now().isoformat(),
    "n_level1_models": len(level2_train_df.select_dtypes(include=[np.number]).columns),
    "level1_model_names": list(level2_train_df.select_dtypes(include=[np.number]).columns),
    "n_level2_models": len(L3_MODEL_NAMES),
    "level2_model_names": L3_MODEL_NAMES,
    "n_feature_subsets": N_FEATURE_SUBSETS,
    "feature_subset_fraction": FEATURE_SUBSET_FRACTION,
    "n_folds": skf.n_splits,
    "train_shape": list(X.shape),
    "test_shape": list(X_test.shape) if X_test is not None else None,
    "scale_pos_weight": scale_pos_weight,
    "undersample_fraction": NO_DAMAGE_DROP_FRACTION,
    "l3_weights": {k: float(v) for k, v in L3_WEIGHTS.items()},
    "l3_best_threshold": float(best_l3_thr),
}
with open(RUN_DIR / "run_metadata.json", "w") as f:
    json.dump(run_meta, f, indent=2)

# 6) OOF predictions (Level 1 + Level 3)
oof_all = level2_train_df.copy()
oof_all["l3_blended"] = l3_oof_probs
oof_all.to_csv(RUN_DIR / "oof_predictions.csv", index=True)

# 7) Test predictions (Level 1 + Level 3)
if level2_test_df is not None:
    test_all = level2_test_df.copy()
    if l3_test_probs is not None:
        test_all["l3_blended"] = l3_test_probs
    test_all.to_csv(RUN_DIR / "test_predictions.csv", index=True)

# 8) Level 3 OOF predictions (Level 2 inputs + blended)
level3_oof_out = level3_train_df.copy()
level3_oof_out["l3_blended"] = l3_oof_probs
level3_oof_out.to_csv(RUN_DIR / "l3_oof_predictions.csv", index=True)

print(f"\nRun saved to: {RUN_DIR}")
print(f"  Models:       {len(list((RUN_DIR / 'models').glob('*.joblib')))} files")
print(f"  Plots:        {len(list((RUN_DIR / 'plots').glob('*.png')))} files")
print(f"  Metadata:     run_metadata.json, level1_metrics.json, level2_metrics.json, level3_metrics.json")
print(f"  Predictions:  oof_predictions.csv, l3_oof_predictions.csv" + (", test_predictions.csv" if level2_test_df is not None else ""))
