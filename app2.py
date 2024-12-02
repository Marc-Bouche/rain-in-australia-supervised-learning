import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Page title
st.title("Rain Prediction: Random Forest Model")

# Sidebar for input
st.sidebar.header("Model Settings")
n_estimators = st.sidebar.slider("Number of Estimators", 50, 200, 100, step=10)
max_depth = st.sidebar.selectbox("Max Depth", [None, 10, 20], index=0)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 5, step=1)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 5, 1, step=1)

# Load Dataset
st.header("Dataset Overview")
file_path = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if file_path is not None:
    data = pd.read_csv(file_path)
    st.write("Data Preview:", data.head())
    st.write("Shape of Data:", data.shape)

    # Feature selection
    if st.checkbox("Train the Random Forest Model"):
        features = data.drop("RainToday", axis=1)  # Replace 'RainToday' with your target column
        target = data["RainToday"]

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        # Model Training
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        rf_model.fit(X_train, y_train)

        # Predictions
        y_pred = rf_model.predict(X_test)

        # Metrics
        st.subheader("Evaluation Metrics")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Oranges", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)