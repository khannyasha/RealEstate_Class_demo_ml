import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor


st.set_page_config(
    page_title="HomeValue AI",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 HomeValue AI")
st.subheader("House Price Prediction Demo App")

st.write(
    """
    This app estimates a house sale price using a machine learning model trained on housing data.
    It is a learning prototype and should not replace a professional valuation.
    """
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "house_price_model_pipeline.pkl"
METADATA_PATH = APP_DIR / "house_price_app_metadata.pkl"
TRAIN_CSV_PATH = APP_DIR / "train.csv"


def make_one_hot_encoder():
    """
    Keeps the app compatible across different scikit-learn versions.
    Newer versions use sparse_output.
    Older versions use sparse.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def add_house_features(data):
    data = data.copy()

    if "YrSold" in data.columns and "YearBuilt" in data.columns:
        data["HouseAge"] = data["YrSold"] - data["YearBuilt"]
    elif "YearBuilt" in data.columns:
        data["HouseAge"] = 2026 - data["YearBuilt"]

    if "YrSold" in data.columns and "YearRemodAdd" in data.columns:
        data["RemodAge"] = data["YrSold"] - data["YearRemodAdd"]
    elif "YearRemodAdd" in data.columns:
        data["RemodAge"] = 2026 - data["YearRemodAdd"]

    square_foot_cols = [col for col in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"] if col in data.columns]
    if square_foot_cols:
        data["TotalSF"] = data[square_foot_cols].sum(axis=1)

    if any(col in data.columns for col in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]):
        data["TotalBath"] = 0
        if "FullBath" in data.columns:
            data["TotalBath"] += data["FullBath"]
        if "HalfBath" in data.columns:
            data["TotalBath"] += 0.5 * data["HalfBath"]
        if "BsmtFullBath" in data.columns:
            data["TotalBath"] += data["BsmtFullBath"]
        if "BsmtHalfBath" in data.columns:
            data["TotalBath"] += 0.5 * data["BsmtHalfBath"]

    if "GarageArea" in data.columns:
        data["HasGarage"] = np.where(data["GarageArea"].fillna(0) > 0, "Yes", "No")

    if "TotalBsmtSF" in data.columns:
        data["HasBasement"] = np.where(data["TotalBsmtSF"].fillna(0) > 0, "Yes", "No")

    return data


def select_model_features(df_fe):
    candidate_numeric_features = [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "GarageArea",
        "TotalBsmtSF",
        "1stFlrSF",
        "FullBath",
        "TotRmsAbvGrd",
        "YearBuilt",
        "YearRemodAdd",
        "LotArea",
        "HouseAge",
        "RemodAge",
        "TotalSF",
        "TotalBath"
    ]

    candidate_categorical_features = [
        "Neighborhood",
        "HouseStyle",
        "ExterQual",
        "KitchenQual",
        "BsmtQual",
        "GarageType",
        "SaleCondition",
        "HasGarage",
        "HasBasement"
    ]

    numeric_features = [col for col in candidate_numeric_features if col in df_fe.columns]
    categorical_features = [col for col in candidate_categorical_features if col in df_fe.columns]
    features = numeric_features + categorical_features

    return numeric_features, categorical_features, features


def safe_numeric_default(series):
    if pd.api.types.is_numeric_dtype(series):
        value = series.median()
        if pd.notna(value):
            return float(value)
    return 0.0


def build_metadata(df_fe, model_name, numeric_features, categorical_features, features):
    return {
        "model_name": model_name,
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "features": list(features),
        "categorical_options": {
            col: sorted([str(x) for x in df_fe[col].dropna().unique().tolist()])
            for col in categorical_features
        },
        "numeric_defaults": {
            col: safe_numeric_default(df_fe[col])
            for col in numeric_features
        }
    }


def train_model_from_csv():
    if not TRAIN_CSV_PATH.exists():
        raise FileNotFoundError(
            "train.csv was not found in the app folder. "
            "Add Kaggle's train.csv to the same GitHub repository folder as app.py."
        )

    df = pd.read_csv(TRAIN_CSV_PATH)

    if "SalePrice" not in df.columns:
        raise ValueError("train.csv must contain the target column SalePrice.")

    df_fe = add_house_features(df)
    numeric_features, categorical_features, features = select_model_features(df_fe)

    if not features:
        raise ValueError("No usable model features were found in train.csv.")

    X = df_fe[features]
    y = df_fe["SalePrice"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = GradientBoostingRegressor(random_state=42)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y)

    metadata = build_metadata(
        df_fe=df_fe,
        model_name="Gradient Boosting trained from train.csv",
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        features=features
    )

    return pipeline, metadata


@st.cache_resource
def load_artifacts():
    """
    Preferred path:
    1. Load the saved model files generated by the notebook.

    Backup path:
    2. If loading the saved .pkl files fails because of a deployment environment mismatch,
       train the model from train.csv inside the deployed app.

    This prevents the public Streamlit app from crashing with an AttributeError.
    """
    saved_model_error = None

    if MODEL_PATH.exists() and METADATA_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            metadata = joblib.load(METADATA_PATH)
            return model, metadata, "saved_model", None
        except Exception as error:
            saved_model_error = f"{type(error).__name__}: {error}"

    model, metadata = train_model_from_csv()
    return model, metadata, "trained_from_train_csv", saved_model_error


try:
    model, metadata, artifact_source, saved_model_error = load_artifacts()
except Exception as error:
    st.error("The app could not load or train the model.")
    st.write(
        """
        Please check that the GitHub repository contains either:
        1. `house_price_model_pipeline.pkl` and `house_price_app_metadata.pkl`, or
        2. Kaggle's `train.csv` file in the same folder as `app.py`.
        """
    )
    with st.expander("Show technical error details"):
        st.exception(error)
    st.stop()


if artifact_source == "saved_model":
    st.success("Model loaded from saved notebook files ✅")
else:
    st.warning(
        "The saved model files could not be loaded in this Streamlit environment, "
        "so the app trained a fresh demo model from train.csv instead."
    )
    if saved_model_error:
        with st.expander("Saved model loading error"):
            st.code(saved_model_error)


numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
categorical_options = metadata["categorical_options"]
numeric_defaults = metadata["numeric_defaults"]

st.sidebar.header("Enter House Details")

user_input = {}

for feature in numeric_features:
    default_value = numeric_defaults.get(feature, 0)

    if "Year" in feature:
        value = st.sidebar.number_input(
            feature,
            min_value=1800,
            max_value=2030,
            value=int(round(default_value)),
            step=1
        )
    elif feature == "OverallQual":
        value = st.sidebar.slider(
            feature,
            min_value=1,
            max_value=10,
            value=int(round(default_value)) if 1 <= int(round(default_value)) <= 10 else 5
        )
    elif feature in ["FullBath", "GarageCars", "TotRmsAbvGrd"]:
        value = st.sidebar.number_input(
            feature,
            min_value=0,
            value=int(round(default_value)),
            step=1
        )
    else:
        value = st.sidebar.number_input(
            feature,
            min_value=0.0,
            value=float(default_value),
            step=100.0
        )

    user_input[feature] = value

for feature in categorical_features:
    options = categorical_options.get(feature, [])

    if not options:
        options = ["Unknown"]

    default_index = 0
    value = st.sidebar.selectbox(feature, options=options, index=default_index)
    user_input[feature] = value

input_df = pd.DataFrame([user_input])

st.write("### Input Summary")
st.dataframe(input_df)

if st.button("Predict House Price"):
    try:
        prediction = model.predict(input_df)[0]

        st.success(f"Estimated Sale Price: ${prediction:,.2f}")

        lower_bound = prediction * 0.90
        upper_bound = prediction * 1.10

        st.info(
            f"Suggested interpretation range: ${lower_bound:,.2f} to ${upper_bound:,.2f}. "
            "This range is only a simple uncertainty guide for demo purposes."
        )

        st.write("### Business Interpretation")
        st.write(
            """
            The model uses property characteristics such as quality, size, garage information,
            age, and location-related features to estimate a likely sale price.
            A real business should validate the model using current local market data before using it.
            """
        )
    except Exception as error:
        st.error("The model could not make a prediction from the selected inputs.")
        with st.expander("Show technical error details"):
            st.exception(error)

st.write("---")
st.caption(
    "HomeValue AI is a teaching demo for Data Science, Machine Learning and Streamlit deployment."
)
