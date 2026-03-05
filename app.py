import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("Middle East Economic Data Analysis")

# File upload
uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Info")
    st.write(df.describe())

    # Select numeric columns
    numeric_df = df.select_dtypes(include=['int64','float64'])

    st.subheader("Numeric Data")
    st.write(numeric_df.head())

    # Target variable
    target = st.selectbox("Select Target Column (Prediction)", numeric_df.columns)

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    score = r2_score(y_test, y_pred)

    st.subheader("Model Accuracy (R2 Score)")
    st.write(score)

    # Visualization
    st.subheader("Prediction vs Actual")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")

    st.pyplot(fig)