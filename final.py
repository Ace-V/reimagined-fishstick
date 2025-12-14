import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

model_file = "../model/water_model.pkl"  

with open(model_file, "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]

csv_file = "../data/Water_data.csv" 
df = pd.read_csv(csv_file)

st.title("💧 Water Quality Classification App")
st.write("Enter water parameters and see predictions + visual analytics.")

st.sidebar.header("Adjust Water Parameters:")

flow = st.sidebar.slider("Flow rate", 
                         min_value=0.0, max_value=250.0, step=0.5)

temp = st.sidebar.slider("Temperature (°C)", 
                         min_value=0.0, max_value=60.0, step=0.1)

turbidity = st.sidebar.slider("Turbidity (NTU)", 
                              min_value=0.0, max_value=250.0, step=0.5)

tds = st.sidebar.slider("TDS (ppm)", 
                        min_value=0.0, max_value=7000.0, step=1.0)

ph = st.sidebar.slider("pH level", 
                       min_value=0.0, max_value=14.0, step=0.1)

if st.sidebar.button("Predict Category"):
    X = scaler.transform([[flow, temp, turbidity, tds, ph]])
    pred = model.predict(X)[0]
    st.success(f"Predicted Water Category: **{pred}**")


st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.subheader("📊 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True, ax=ax)
st.pyplot(fig)

st.subheader("📈 Category-wise Average Values")

fig2, ax2 = plt.subplots(figsize=(8, 5))
df.groupby("category").mean(numeric_only=True).plot(kind="bar", ax=ax2)
st.pyplot(fig2)

st.subheader("📌 Feature Distributions")

features = ["flow", "temperature", "turbidity", "tds", "ph"]

for col in features:
    fig3, ax3 = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax3)
    ax3.set_title(f"Distribution of {col}")
    st.pyplot(fig3)

st.subheader("📌 Pairplot (Feature Relationships)")
st.write("⚠ This may take a few seconds to load.")

pairplot_fig = sns.pairplot(df, hue="category")
st.pyplot(pairplot_fig)
