import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATA_PATH = "data/sample_dataset.csv"
FEATURE_COLUMNS = [
    "attendance_percentage",
    "internal_marks",
    "assignment_delay_days",
]
TARGET_COLUMN = "dropout_risk"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    # Fill missing numeric values with the median of each column
    for column in FEATURE_COLUMNS:
        df[column] = df[column].fillna(df[column].median())

    # Fill missing target labels (if any) with the most frequent label
    if df[TARGET_COLUMN].isna().any():
        df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(df[TARGET_COLUMN].mode()[0])

    return df


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset from CSV and handle missing values."""
    df = pd.read_csv(csv_path)
    return clean_dataset(df)


def preprocess_data(df: pd.DataFrame):
    """Normalize features and encode target labels."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Encode labels like Low/Medium/High into numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Normalize numeric features so the model trains more consistently
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, scaler, label_encoder


def train_model(X: np.ndarray, y: np.ndarray):
    """Split data, train model, and return metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


def predict_dropout_risk(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    attendance: float,
    marks: float,
    delay_days: float,
) -> str:
    """Predict dropout risk label for a single student."""
    input_data = np.array([[attendance, marks, delay_days]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return label_encoder.inverse_transform([prediction])[0]


def predict_batch(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict dropout risk for multiple students and return an updated DataFrame."""
    features_scaled = scaler.transform(df[FEATURE_COLUMNS])
    predictions = model.predict(features_scaled)
    df_with_predictions = df.copy()
    df_with_predictions["predicted_risk"] = label_encoder.inverse_transform(predictions)
    return df_with_predictions


def display_feature_importance(model: RandomForestClassifier):
    """Show which features influence dropout risk most."""
    importance_series = pd.Series(
        model.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    st.subheader("Feature Importance")
    st.write(
        "Random Forest ranks features by how much they reduce prediction errors. "
        "Higher bars indicate stronger influence on dropout risk."
    )
    st.bar_chart(importance_series)

    # Comment on the top drivers for explainability
    st.caption(
        "Typically, lower attendance and higher assignment delays increase risk, "
        "while higher internal marks lower risk."
    )


def main():
    st.set_page_config(page_title="EduGuard AI", layout="wide")
    st.title("EduGuard AI – Early Dropout Risk Prediction System")

    st.write(
        "This hackathon MVP predicts dropout risk using attendance, marks, "
        "and assignment delays with a Random Forest model."
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV file (optional)", type=["csv"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        df = clean_dataset(pd.read_csv(uploaded_file))
    else:
        df = load_dataset(DATA_PATH)

    X_scaled, y_encoded, scaler, label_encoder = preprocess_data(df)
    model, accuracy = train_model(X_scaled, y_encoded)

    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    tab_student, tab_faculty = st.tabs(["Student Dashboard", "Faculty Dashboard"])

    with tab_student:
        st.header("Student Dashboard")
        st.write("Enter student details and predict dropout risk.")

        attendance = st.number_input(
            "Attendance Percentage",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
        )
        marks = st.number_input(
            "Internal Marks",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
        )
        delay_days = st.number_input(
            "Assignment Delay (days)",
            min_value=0.0,
            max_value=30.0,
            value=2.0,
            step=1.0,
        )

        if st.button("Predict Dropout Risk"):
            risk_label = predict_dropout_risk(
                model, scaler, label_encoder, attendance, marks, delay_days
            )
            st.success(f"Predicted Dropout Risk: {risk_label}")

    with tab_faculty:
        st.header("Faculty Dashboard")
        st.write("Predicted risk levels for the uploaded dataset.")

        predictions_df = predict_batch(model, scaler, label_encoder, df)
        st.dataframe(predictions_df, use_container_width=True)

        st.subheader("Risk Distribution")
        risk_counts = predictions_df["predicted_risk"].value_counts().sort_index()
        st.bar_chart(risk_counts)

        display_feature_importance(model)


if __name__ == "__main__":
    # Quick console run for sanity checks (prints accuracy and a sample prediction)
    dataset = load_dataset(DATA_PATH)
    X_scaled_data, y_encoded_data, scaler_data, label_encoder_data = preprocess_data(
        dataset
    )
    trained_model, model_accuracy = train_model(X_scaled_data, y_encoded_data)

    sample_prediction = predict_dropout_risk(
        trained_model,
        scaler_data,
        label_encoder_data,
        attendance=82,
        marks=74,
        delay_days=2,
    )

    print(f"Model accuracy: {model_accuracy * 100:.2f}%")
    print(f"Sample student prediction: {sample_prediction}")

    # Run the Streamlit UI
    main()
