"""EduGuard AI - Early Dropout Risk Prediction System (Streamlit MVP)."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

DATA_PATH = "data/student_data.csv"
FEATURE_COLUMNS = [
    "attendance_percentage",
    "internal_marks",
    "assignment_delay_days",
]
TARGET_COLUMN = "dropout_risk"


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the dataset from CSV."""
    return pd.read_csv(csv_path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with the median for each column."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    return df


def normalize_features(features: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Normalize numeric features using StandardScaler."""
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_values, columns=features.columns)
    return scaled_df, scaler


def encode_labels(labels: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    """Encode categorical labels into numeric values."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return pd.Series(encoded, name=labels.name), encoder


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train Random Forest model and return artifacts for inference."""
    df = handle_missing_values(df.copy())
    features = df[FEATURE_COLUMNS]
    labels = df[TARGET_COLUMN]

    scaled_features, scaler = normalize_features(features)
    encoded_labels, label_encoder = encode_labels(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        scaled_features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    model = RandomForestClassifier(random_state=42, n_estimators=200)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, scaler, label_encoder, accuracy


def predict_risk(model, scaler, label_encoder, input_data: dict) -> str:
    """Predict dropout risk for a single student."""
    input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    return label_encoder.inverse_transform(prediction)[0]


def get_feature_importance(model) -> pd.DataFrame:
    """Return feature importance as a DataFrame for visualization."""
    importance = model.feature_importances_
    return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importance}).sort_values(
        by="importance", ascending=False
    )


def student_dashboard(model, scaler, label_encoder):
    """Render the student dashboard for single prediction."""
    st.subheader("Student Dashboard")
    attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0, value=85.0)
    marks = st.number_input("Internal Marks", min_value=0.0, max_value=100.0, value=70.0)
    delay = st.number_input("Assignment Delay Days", min_value=0, max_value=30, value=3)

    if st.button("Predict Dropout Risk"):
        risk = predict_risk(
            model,
            scaler,
            label_encoder,
            {
                "attendance_percentage": attendance,
                "internal_marks": marks,
                "assignment_delay_days": delay,
            },
        )
        st.success(f"Predicted Dropout Risk: **{risk}**")


def faculty_dashboard(df: pd.DataFrame, model, scaler, label_encoder):
    """Render the faculty dashboard for multiple predictions."""
    st.subheader("Faculty Dashboard")

    prediction_inputs = df[FEATURE_COLUMNS].copy()
    scaled_inputs = scaler.transform(prediction_inputs)
    predicted_labels = model.predict(scaled_inputs)
    df_with_predictions = df.copy()
    df_with_predictions["predicted_risk"] = label_encoder.inverse_transform(predicted_labels)

    st.write("### Student Risk Table")
    st.dataframe(df_with_predictions)

    st.write("### Risk Distribution")
    risk_counts = df_with_predictions["predicted_risk"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]
    st.bar_chart(risk_counts.set_index("risk_level"))


def show_feature_importance(feature_importance: pd.DataFrame):
    """Display feature importance with a note about influence."""
    st.write("### Feature Importance")
    st.dataframe(feature_importance)
    st.caption(
        "Higher importance means the feature has more influence on the dropout risk prediction."
    )


def main():
    """Main Streamlit application."""
    st.title("EduGuard AI – Early Dropout Risk Prediction System")
    st.write(
        "This demo uses attendance, internal marks, and assignment delay days to predict "
        "dropout risk (Low, Medium, High) using a Random Forest Classifier."
    )

    data = load_data(DATA_PATH)
    model, scaler, label_encoder, accuracy = train_model(data)

    st.sidebar.write("### Model Accuracy")
    st.sidebar.write(f"Accuracy on test set: **{accuracy:.2f}**")

    tabs = st.tabs(["Student Dashboard", "Faculty Dashboard", "Explainability"])

    with tabs[0]:
        student_dashboard(model, scaler, label_encoder)

    with tabs[1]:
        faculty_dashboard(data, model, scaler, label_encoder)

    with tabs[2]:
        feature_importance = get_feature_importance(model)
        show_feature_importance(feature_importance)
        # Typically, lower attendance and higher assignment delays push risk higher.
        # Internal marks also contribute: lower marks often correlate with higher risk.


if __name__ == "__main__":
    main()
