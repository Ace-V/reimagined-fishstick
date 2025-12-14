import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
model_file = "../model/water_safety_models.pkl"  

with open(model_file, "rb") as f:
    models_dict = pickle.load(f)

# Load dataset
csv_file = "../data/Water_data.csv" 
df = pd.read_csv(csv_file)

# Get available categories
categories = list(models_dict.keys())

st.title("💧 Water Safety Classification App")
st.write("Select your water category and check if the water parameters are safe for that use.")

# Category selection
st.sidebar.header("Step 1: Select Water Category")
selected_category = st.sidebar.selectbox(
    "What is this water intended for?",
    options=categories
)

st.sidebar.markdown("---")
st.sidebar.header("Step 2: Enter Water Parameters")

# Parameter inputs
flow = st.sidebar.slider("Flow rate", 
                         min_value=0.0, max_value=250.0, step=0.5, value=100.0)

temp = st.sidebar.slider("Temperature (°C)", 
                         min_value=0.0, max_value=60.0, step=0.1, value=25.0)

turbidity = st.sidebar.slider("Turbidity (NTU)", 
                              min_value=0.0, max_value=250.0, step=0.5, value=50.0)

tds = st.sidebar.slider("TDS (ppm)", 
                        min_value=0.0, max_value=7000.0, step=1.0, value=500.0)

ph = st.sidebar.slider("pH level", 
                       min_value=0.0, max_value=14.0, step=0.1, value=7.0)

# Prediction
if st.sidebar.button("Check Water Safety", type="primary"):
    # Get the model and scaler for selected category
    model = models_dict[selected_category]["model"]
    scaler = models_dict[selected_category]["scaler"]
    
    # Transform input and predict
    X = scaler.transform([[flow, temp, turbidity, tds, ph]])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    # Display result
    st.markdown("---")
    st.subheader(f"🎯 Safety Check for: **{selected_category}** Water")
    
    if prediction == 1:
        st.success(f"✅ **SAFE** - Water parameters are suitable for {selected_category} use")
        st.metric("Confidence", f"{probability[1]*100:.1f}%")
    else:
        st.error(f"❌ **UNSAFE** - Water parameters are NOT suitable for {selected_category} use")
        st.metric("Confidence", f"{probability[0]*100:.1f}%")
    
    # Show input parameters
    with st.expander("📋 Your Input Parameters"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Flow", f"{flow}")
        col1.metric("Temperature", f"{temp}°C")
        col2.metric("Turbidity", f"{turbidity} NTU")
        col2.metric("TDS", f"{tds} ppm")
        col3.metric("pH", f"{ph}")

# Dataset insights
st.markdown("---")
st.subheader("📊 Dataset Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["📄 Data Preview", "🔥 Heatmap", "📈 Category Averages", "📉 Distributions"])

with tab1:
    st.dataframe(df.head(10))
    st.caption(f"Total samples: {len(df)}")

with tab2:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

with tab3:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    category_means = df.groupby("category").mean(numeric_only=True)
    category_means.plot(kind="bar", ax=ax2)
    ax2.set_title("Average Parameter Values by Category")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Average Value")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

with tab4:
    features = ["flow", "temperature", "turbidity", "tds", "ph"]
    
    cols = st.columns(2)
    for idx, col in enumerate(features):
        with cols[idx % 2]:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.histplot(df[col], kde=True, ax=ax3)
            ax3.set_title(f"Distribution of {col.capitalize()}")
            st.pyplot(fig3)

# Category-specific statistics
st.markdown("---")
st.subheader(f"📌 {selected_category} Water - Typical Ranges")

category_data = df[df['category'] == selected_category]
if len(category_data) > 0:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Flow (avg)", f"{category_data['flow'].mean():.1f}")
    col2.metric("Temp (avg)", f"{category_data['temperature'].mean():.1f}°C")
    col3.metric("Turbidity (avg)", f"{category_data['turbidity'].mean():.1f}")
    col4.metric("TDS (avg)", f"{category_data['tds'].mean():.0f}")
    col5.metric("pH (avg)", f"{category_data['ph'].mean():.2f}")
    
    with st.expander("📊 See detailed statistics"):
        st.dataframe(category_data.describe())
else:
    st.info(f"No data available for {selected_category} category")