import streamlit as st
import pandas as pd
import pathlib
import sqlite3
import tempfile
import PyPDF2
import docx
import hashlib

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
        st.session_state['seen_hashes'] = {} 
        
    file_names = [file.name for file in uploaded_files]
    tabs = st.tabs(file_names)
    
    current_batch_names = set()
    
    for file, tab in zip(uploaded_files, tabs):
        with tab:
            # --- 1. NAME DUPLICATE CATCHER ---
            if file.name in current_batch_names:
                st.toast(f"Duplicate Name: {file.name}", icon="🚨")
                st.error(f"📄 **{file.name}**\n\n🚨 **DUPLICATE NAME:** Uploaded multiple times in this batch. Skipping.")
                continue
            current_batch_names.add(file.name)

            # --- 2. CONTENT DUPLICATE CATCHER (The Popup & Red Alert) ---
            file_hash = hashlib.md5(file.getvalue()).hexdigest()
            
            if file_hash in st.session_state['seen_hashes']:
                original_name = st.session_state['seen_hashes'][file_hash]
                if original_name != file.name:
                    # The Pop-up!
                    st.toast(f"Duplicate DNA caught: {file.name}", icon="🚫")
                    # The Red Box Alert!
                    st.error(f"📄 **{file.name}**\n\n🚨 **DUPLICATE DATA:** Exact same contents as '{original_name}'. Skipping.")
                    continue
            else:
                st.session_state['seen_hashes'][file_hash] = file.name
            
            # --- FILE PROCESSING ---
            file_extension = pathlib.Path(file.name).suffix.lower()
            
            try:
                # --- STRUCTURED DATA (With UI fix) ---
                if file_extension in ['.csv', '.xlsx', '.xls', '.json']:
                    if file_extension == '.csv':
                        df = pd.read_csv(file)
                    elif file_extension in ['.xlsx', '.xls']:
                        df = pd.read_excel(file)
                    elif file_extension == '.json':
                        df = pd.read_json(file, orient='records')
                        
                    st.session_state['datasets'][file.name] = df
                    st.success(f"Loaded Table: {file.name}")
                    # UI FIX: Lock height and force it to fit the container
                    st.dataframe(df.head(50), use_container_width=True, height=300)
                
                # --- SQL DATABASES (With UI fix) ---
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
                        # UI FIX
                        st.dataframe(df.head(50), use_container_width=True, height=300)
                    conn.close()

                # --- UNSTRUCTURED DATA (With UI fix) ---
                elif file_extension == '.txt':
                    text_data = file.getvalue().decode("utf-8")
                    st.session_state['documents'][file.name] = text_data
                    st.success(f"Loaded Text Document: {file.name}")
                    # UI FIX: Lock height so it doesn't stretch the page
                    st.text_area("Preview", text_data[:1000] + "...", height=250, key=f"preview_{file.name}")
                    
                elif file_extension == '.pdf':
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_data = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
                    st.session_state['documents'][file.name] = text_data
                    st.success(f"Loaded PDF: {file.name}")
                    st.text_area("Preview", text_data[:1000] + "...", height=250, key=f"preview_{file.name}")
                    
                elif file_extension == '.docx':
                    doc = docx.Document(file)
                    text_data = "\n".join([para.text for para in doc.paragraphs])
                    st.session_state['documents'][file.name] = text_data
                    st.success(f"Loaded Word Doc: {file.name}")
                    st.text_area("Preview", text_data[:1000] + "...", height=250, key=f"preview_{file.name}")

            except Exception as e:
                st.error(f"🚨 FILE CORRUPTED OR UNSUPPORTED: Could not read '{file.name}'.")
                st.write(f"Error code: {e}")
