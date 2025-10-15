import streamlit as st
import pandas as pd
import pickle

# ------------------------------------------
# 1️⃣ Load Model and Encoders
# ------------------------------------------
MODEL_PATH = "attrition_model.pkl"
ENCODER_PATH = "label_encoders.pkl"
DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Load model and encoders
with open(MODEL_PATH, "rb") as model_file:
    rf_model = pickle.load(model_file)
with open(ENCODER_PATH, "rb") as le_file:
    label_encoders = pickle.load(le_file)

df = pd.read_csv(DATA_PATH)
df_encoded = df.copy()
for col, le in label_encoders.items():
    df_encoded[col] = le.transform(df[col])
X = df_encoded.drop("Attrition", axis=1)

# ------------------------------------------
# 2️⃣ Streamlit Page Setup
# ------------------------------------------
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("🧠 Employee Attrition Prediction")
st.markdown("---")

# ------------------------------------------
# 3️⃣ User Input Section
# ------------------------------------------
skip_columns = ["EmployeeCount", "Over18", "StandardHours"]
cols = [c for c in X.columns if c not in skip_columns]

st.sidebar.header("📋 Input Employee Details")

user_data = {}

for col in cols:
    if col in label_encoders:
        options = list(label_encoders[col].classes_)
        default_idx = 0
        user_data[col] = st.sidebar.selectbox(col, options, index=default_idx)
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_data[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

# ------------------------------------------
# 4️⃣ Prediction
# ------------------------------------------
if st.sidebar.button("🚀 Predict Attrition"):
    for col in skip_columns:
        user_data[col] = df_encoded[col].iloc[0]

    # Encode categorical
    for col in label_encoders:
        if col in user_data:
            user_data[col] = label_encoders[col].transform([user_data[col]])[0]

    user_df = pd.DataFrame([user_data])
    prediction = rf_model.predict(user_df)[0]
    pred_label = label_encoders["Attrition"].inverse_transform([prediction])[0]

    if pred_label == "Yes":
        st.error("⚠️ This employee is **likely to leave** (Attrition: YES)")
    else:
        st.success("✅ This employee is **likely to stay** (Attrition: NO)")

# ------------------------------------------
# 5️⃣ Footer
# ------------------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit")
