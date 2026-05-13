import streamlit as st
import pandas as pd
import pathlib
import sqlite3
import tempfile
import PyPDF2
import docx
import hashlib # NEW: This is our file "DNA" scanner

st.set_page_config(page_title="AI Data Agent", layout="wide")

st.title("🗂️ The Ultimate Data & Doc Loader")
st.write("Upload CSV, Excel, JSON, SQL (.db), PDF, Word (.docx), or TXT files.")

uploaded_files = st.file_uploader(
    "Drop your files here", 
    type=["csv", "xlsx", "xls", "json", "db", "sqlite", "pdf", "docx", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    if 'datasets' not in st.session_state:
        st.session_state['datasets'] = {} 
    if 'documents' not in st.session_state:
        st.session_state['documents'] = {} 
    if 'seen_hashes' not in st.session_state:
        st.session_state['seen_hashes'] = {} # Memory for file DNA
        
    file_names = [file.name for file in uploaded_files]
    tabs = st.tabs(file_names)
    
    current_batch_names = set()
    
    for file, tab in zip(uploaded_files, tabs):
        with tab:
            # --- 1. NAME DUPLICATE CATCHER (If uploaded at the exact same time) ---
            if file.name in current_batch_names:
                st.error(f"🚨 DUPLICATE NAME: You uploaded '{file.name}' more than once in this batch.")
                continue
            current_batch_names.add(file.name)

            # --- 2. CONTENT DUPLICATE CATCHER (The DNA Scanner) ---
            file_hash = hashlib.md5(file.getvalue()).hexdigest()
            
            # If we've seen this exact data before, but under a different name
            if file_hash in st.session_state['seen_hashes']:
                original_name = st.session_state['seen_hashes'][file_hash]
                if original_name != file.name:
                    st.error(f"🚨 DUPLICATE DATA DETECTED: '{file.name}' has the exact same contents as '{original_name}'.")
                    st.warning("Skipping this file to prevent overlapping data in the system.")
                    continue
            else:
                # Add this new file's DNA to our long-term memory
                st.session_state['seen_hashes'][file_hash] = file.name
            
            # --- FILE PROCESSING ---
            file_extension = pathlib.Path(file.name).suffix.lower()
            
            try:
                # --- STRUCTURED DATA ---
                if file_extension in ['.csv', '.xlsx', '.xls', '.json']:
                    if file_extension == '.csv':
                        df = pd.read_csv(file)
                    elif file_extension in ['.xlsx', '.xls']:
                        df = pd.read_excel(file)
                    elif file_extension == '.json':
                        df = pd.read_json(file, orient='records')
                        
                    st.session_state['datasets'][file.name] = df
                    st.success(f"Loaded Table: {file.name}")
                    st.dataframe(df.head())
                
                # --- SQL DATABASES ---
                elif file_extension in ['.db', '.sqlite']:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                        
                    conn = sqlite3.connect(tmp_path)
                    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
                    
                    st.success(f"Connected to SQL Database: {file.name}")
                    st.write("Tables found:", tables['name'].tolist())
                    
                    if not tables.empty:
                        first_table = tables['name'].iloc[0]
                        df = pd.read_sql_query(f"SELECT * FROM {first_table}", conn)
                        st.session_state['datasets'][f"{file.name}_{first_table}"] = df
                        st.write(f"Preview of `{first_table}`:")
                        st.dataframe(df.head())
                    conn.close()

                # --- UNSTRUCTURED DATA ---
                elif file_extension == '.txt':
                    text_data = file.getvalue().decode("utf-8")
                    st.session_state['documents'][file.name] = text_data
                    st.success(f"Loaded Text Document: {file.name}")
                    st.text_area("Preview", text_data[:500] + "...", height=150, key=f"preview_{file.name}")
                    
                elif file_extension == '.pdf':
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_data = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
                    st.session_state['documents'][file.name] = text_data
                    st.success(f"Loaded PDF: {file.name}")
                    st.text_area("Preview", text_data[:500] + "...", height=150, key=f"preview_{file.name}")
                    
                elif file_extension == '.docx':
                    doc = docx.Document(file)
                    text_data = "\n".join([para.text for para in doc.paragraphs])
                    st.session_state['documents'][file.name] = text_data
                    st.success(f"Loaded Word Doc: {file.name}")
                    st.text_area("Preview", text_data[:500] + "...", height=150, key=f"preview_{file.name}")

            except Exception as e:
                st.error(f"🚨 FILE CORRUPTED OR UNSUPPORTED: Could not read '{file.name}'.")
                st.write(f"Error code: {e}")
