import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np

# Load models
model_file = "./model/water_safety_models.pkl"  

with open(model_file, "rb") as f:
    models_dict = pickle.load(f)

# Load dataset
csv_file = "../data/Water_data.csv" 
df = pd.read_csv(csv_file)

# Get available categories
categories = list(models_dict.keys())

st.title("💧 Water Safety Classification App with AI Explainability")
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
    input_data = [flow, temp, turbidity, tds, ph]
    X = scaler.transform([input_data])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    # Display result
    st.markdown("---")
    st.subheader(f"🎯 Safety Check for: **{selected_category}** Water")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if prediction == 1:
            st.success(f"✅ **SAFE** - Water parameters are suitable for {selected_category} use")
        else:
            st.error(f"❌ **UNSAFE** - Water parameters are NOT suitable for {selected_category} use")
    
    with col2:
        st.metric("Confidence", f"{max(probability)*100:.1f}%")
    
    # Show input parameters
    with st.expander("📋 Your Input Parameters"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Flow", f"{flow}")
        col1.metric("Temperature", f"{temp}°C")
        col2.metric("Turbidity", f"{turbidity} NTU")
        col2.metric("TDS", f"{tds} ppm")
        col3.metric("pH", f"{ph}")
    
    # ==================== SHAP EXPLANATION ====================
    st.markdown("---")
    st.subheader("🔍 AI Explanation: Why This Decision?")
    st.write("Understanding which parameters influenced the safety prediction:")
    
    # Create SHAP explainer
    with st.spinner("Generating AI explanation..."):
        # For tree-based models, use TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for this prediction
        shap_values = explainer.shap_values(X)
        
        # For binary classification, get values for positive class (safe)
        if isinstance(shap_values, list):
            shap_values_display = shap_values[1]  # Class 1 (safe)
        else:
            shap_values_display = shap_values
        
        # Get the SHAP values for our single prediction
        shap_values_single = shap_values_display[0]
        
        # Convert all SHAP values to plain Python floats to avoid numpy array issues
        shap_values_list = []
        for val in shap_values_single:
            if isinstance(val, np.ndarray):
                shap_values_list.append(float(val.item() if val.size == 1 else val.flatten()[0]))
            else:
                shap_values_list.append(float(val))
        
        # Feature names
        feature_names = ['Flow', 'Temperature', 'Turbidity', 'TDS', 'pH']
        
        # ========== 1. Feature Contribution Table ==========
        st.write("### 📊 Feature Impact Analysis")
        
        # Create contribution dataframe
        contributions = []
        for i, (feature, value, shap_val) in enumerate(zip(feature_names, input_data, shap_values_list)):
            contributions.append({
                'Feature': feature,
                'Your Value': f"{value:.2f}",
                'Impact Score': shap_val,
                'Direction': '🟢 Safer' if shap_val > 0 else '🔴 Riskier' if shap_val < 0 else '⚪ Neutral'
            })
        
        # Sort by absolute impact
        contributions_df = pd.DataFrame(contributions)
        contributions_df['Abs Impact'] = contributions_df['Impact Score'].abs()
        contributions_df = contributions_df.sort_values('Abs Impact', ascending=False)
        
        # Display table
        st.dataframe(
            contributions_df[['Feature', 'Your Value', 'Impact Score', 'Direction']],
            use_container_width=True,
            hide_index=True
        )
        
        # ========== 2. Visual Bar Chart ==========
        st.write("### 📈 Visual Impact Breakdown")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Convert to numpy array for sorting
        shap_array = np.array(shap_values_list)
        
        # Sort features by SHAP value
        sorted_idx = np.argsort(np.abs(shap_array))[::-1]
        sorted_features = [feature_names[int(i)] for i in sorted_idx]
        sorted_values = shap_array[sorted_idx]
        sorted_inputs = [input_data[int(i)] for i in sorted_idx]
        
        # Create color map
        colors = ['#2ecc71' if val > 0 else '#e74c3c' for val in sorted_values]
        
        # Create horizontal bar chart
        bars = ax.barh(range(len(sorted_features)), sorted_values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels([f"{feat}\n({val:.2f})" for feat, val in zip(sorted_features, sorted_inputs)])
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
        ax.set_title('How Each Parameter Affected the Safety Decision', fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sorted_values)):
            label_x = val + (0.01 if val > 0 else -0.01)
            ha = 'left' if val > 0 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', 
                   va='center', ha=ha, fontweight='bold', fontsize=10)
        
        st.pyplot(fig)
        
        # ========== 3. Interpretation Guide ==========
        st.write("### 💡 How to Read This")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **🟢 Positive Values (Green)**
            - Push prediction toward SAFE
            - These parameters are helping
            - Higher = stronger positive effect
            """)
        
        with col2:
            st.warning("""
            **🔴 Negative Values (Red)**
            - Push prediction toward UNSAFE
            - These parameters are concerning
            - Lower = stronger negative effect
            """)
        
        # ========== 4. Key Insights ==========
        st.write("### 🎯 Key Insights")
        
        # Use the already converted list
        shap_array = np.array(shap_values_list)
        
        # Find most influential features
        most_positive_idx = int(np.argmax(shap_array))
        most_negative_idx = int(np.argmin(shap_array))
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            if shap_values_list[most_positive_idx] > 0:
                st.success(f"""
                **✅ Strongest Positive Factor:**
                - **{feature_names[most_positive_idx]}** = {input_data[most_positive_idx]:.2f}
                - Impact: +{shap_values_list[most_positive_idx]:.3f}
                - This parameter is helping make the water safer
                """)
        
        with insight_col2:
            if shap_values_list[most_negative_idx] < 0:
                st.error(f"""
                **⚠️ Strongest Negative Factor:**
                - **{feature_names[most_negative_idx]}** = {input_data[most_negative_idx]:.2f}
                - Impact: {shap_values_list[most_negative_idx]:.3f}
                - This is the main concern for safety
                """)
        
        # ========== 5. Base Value Explanation ==========
        with st.expander("🤓 Advanced: Understanding Base Values"):
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                if isinstance(base_value, list):
                    base_value = base_value[1]
                else:
        # Handle numpy array case
                    base_value = base_value[1] if base_value.size > 1 else base_value.item()
            
            # Convert shap_array to proper numpy array for sum
            shap_sum = float(np.sum(shap_array))
            final_value = float(base_value) + shap_sum
            
            st.write(f"""
            **How the model thinks:**
            
            1. **Base prediction** (before seeing your data): {base_value:.3f}
               - This is the average prediction across all {selected_category} water samples
            
            2. **Your parameter adjustments**: {shap_sum:+.3f}
               - Sum of all individual SHAP values shown above
            
            3. **Final prediction score**: {final_value:.3f}
               - Base + Adjustments = {base_value:.3f} + {shap_sum:+.3f}
               - This score is converted to Safe/Unsafe decision
            
            **Interpretation:**
            - Score > 0 typically means SAFE
            - Score < 0 typically means UNSAFE
            - Your score: {final_value:.3f} → {"✅ SAFE" if prediction == 1 else "❌ UNSAFE"}
            """)

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
