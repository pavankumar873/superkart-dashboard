import streamlit as st
import pandas as pd
import joblib
import os

# Fix for scikit-learn version mismatch on Streamlit Cloud
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list): 
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# Now you can safely import joblib and your other modules below
import joblib
import streamlit as st
# ... rest of your code
# Set up page configurations
st.set_page_config(page_title="SuperKart Sales Forecaster", layout="centered", page_icon="🛒")

st.title("🛒 SuperKart Retail Sales Forecasting")
st.write("Enter the product and store attributes below to project total revenue.")

# 1. Load the model pipeline safely using caching so it doesn't reload on every click
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.joblib. Make sure the file is in the same directory. Details: {e}")

# 2. Build the interactive form UI
with st.form("prediction_form"):
    st.subheader("Product Details")
    col1, col2 = st.columns(2)
    with col1:
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0, step=1.0)
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5, step=0.1)
    with col2:
        product_sugar = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_type = st.selectbox("Product Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Baking Goods", "Snack Foods", "Household"])
        
    product_area = st.number_input("Allocated Area (Visibility Ratio)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

    st.subheader("Store Details")
    col3, col4 = st.columns(2)
    with col3:
        store_id = st.selectbox("Store ID", ["OUT010", "OUT013", "OUT027", "OUT046", "OUT049"])
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    with col4:
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])
        store_city = st.selectbox("City Location Type", ["Tier 1", "Tier 2", "Tier 3"])
        
    store_year = st.number_input("Establishment Year", min_value=1900, max_value=2026, value=2010, step=1)

    submit = st.form_submit_button("Predict Future Sales")

# 3. Handle prediction logic when user clicks the button
if submit:
    # Construct input dictionary mirroring your exact training feature names
    input_data = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar,
        "Product_Allocated_Area": product_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Id": store_id,
        "Store_Establishment_Year": int(store_year),
        "Store_Size": store_size,
        "Store_Location_City_Type": store_city,
        "Store_Type": store_type
    }
    
    # Convert to DataFrame for the scikit-learn/XGBoost pipeline
    input_df = pd.DataFrame([input_data])
    
    # Run pipeline inference
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🎯 **Projected Store Revenue for Product:** ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed. Ensure feature names match your training data. Error: {e}")
