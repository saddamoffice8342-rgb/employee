import streamlit as st
import pandas as pd
import pickle

# =====================================================
# 1️⃣ Load Model and Encoders
# =====================================================
MODEL_PATH = "attrition_model.pkl"
ENCODER_PATH = "label_encoders.pkl"
DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Load model and encoders
with open(MODEL_PATH, "rb") as model_file:
    rf_model = pickle.load(model_file)
with open(ENCODER_PATH, "rb") as le_file:
    label_encoders = pickle.load(le_file)

# Load dataset
df = pd.read_csv(DATA_PATH)
df_encoded = df.copy()
for col, le in label_encoders.items():
    df_encoded[col] = le.transform(df[col])
X = df_encoded.drop("Attrition", axis=1)

# =====================================================
# 2️⃣ Streamlit Page Setup
# =====================================================
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("🧠 Employee Attrition Prediction")
st.markdown("---")

# =====================================================
# 3️⃣ Input Section (Sidebar)
# =====================================================
st.sidebar.header("📋 Enter Employee Details")

skip_columns = ["EmployeeCount", "Over18", "StandardHours"]
cols = [c for c in X.columns if c not in skip_columns]

user_data = {}

for col in cols:
    if col in label_encoders:
        # Dropdown for categorical columns
        options = list(label_encoders[col].classes_)
        default_idx = 0
        user_data[col] = st.sidebar.selectbox(col, options, index=default_idx)
    else:
        # Slider for numeric columns
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_data[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

# =====================================================
# 4️⃣ Prediction Button
# =====================================================
if st.sidebar.button("🚀 Predict Attrition"):
    # Add constant columns
    for col in skip_columns:
        user_data[col] = df_encoded[col].iloc[0]

    # Safely encode categorical columns
    for col, le in label_encoders.items():
        if col in user_data:
            try:
                user_data[col] = le.transform([user_data[col]])[0]
            except ValueError:
                st.warning(f"⚠️ Unexpected category '{user_data[col]}' in '{col}', using default.")
                user_data[col] = le.transform([le.classes_[0]])[0]

    # Create DataFrame for prediction
    user_df = pd.DataFrame([user_data])

    # 🧩 Ensure columns exactly match training features
    user_df = user_df.reindex(columns=X.columns, fill_value=0)

    # ✅ Predict
    prediction = rf_model.predict(user_df)[0]
    pred_label = label_encoders["Attrition"].inverse_transform([prediction])[0]

    # =====================================================
    # 5️⃣ Display Result
    # =====================================================
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if pred_label == "Yes":
        st.error("⚠️ This employee is **likely to leave** (Attrition: YES)")
    else:
        st.success("✅ This employee is **likely to stay** (Attrition: NO)")

# =====================================================
# 6️⃣ Footer
# =====================================================
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit")

