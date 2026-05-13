import streamlit as st
import pandas as pd
import pathlib

# Makes the app wider and look nicer
st.set_page_config(page_title="AI Data Agent", layout="wide")

st.title("🗂️ Universal Data Loader")
st.write("Upload a CSV, Excel, or JSON file to get started.")

# 1. Create the drag-and-drop uploader box
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls", "json"])

# 2. Logic to handle the file once it's uploaded
if uploaded_file is not None:
    # Figure out what kind of file it is (e.g., '.csv')
    file_extension = pathlib.Path(uploaded_file.name).suffix.lower()
    
    try:
        # Use pandas to read the file based on its type
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == '.json':
            df = pd.read_json(uploaded_file, orient='records')
        
        # Save the data in Streamlit's "memory" so we can use it in later steps
        st.session_state['raw_data'] = df
        
        # 3. Display the results!
        st.success(f"Success! Loaded {uploaded_file.name}")
        st.write("Here is a quick preview of your data:")
        st.dataframe(df.head()) # This displays the first 5 rows nicely
        
    except Exception as e:
        # If it crashes, show a nice error message instead of a red screen of death
        st.error(f"Oops! Something went wrong loading the file: {e}")
