import streamlit as st
import pandas as pd
import pathlib
import sqlite3
import tempfile
import PyPDF2
import docx

st.set_page_config(page_title="AI Data Agent", layout="wide")

st.title("🗂️ The Ultimate Data & Doc Loader")
st.write("Upload CSV, Excel, JSON, SQL (.db), PDF, Word (.docx), or TXT files.")

# 1. Expanded file types!
uploaded_files = st.file_uploader(
    "Drop your files here", 
    type=["csv", "xlsx", "xls", "json", "db", "sqlite", "pdf", "docx", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    # We now have TWO memory boxes: One for Tables, one for Text
    if 'datasets' not in st.session_state:
        st.session_state['datasets'] = {} 
    if 'documents' not in st.session_state:
        st.session_state['documents'] = {} 
        
    file_names = [file.name for file in uploaded_files]
    tabs = st.tabs(file_names)
    
    for file, tab in zip(uploaded_files, tabs):
        file_extension = pathlib.Path(file.name).suffix.lower()
        
        with tab:
            try:
                # --- STRUCTURED DATA (Spreadsheets) ---
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
                
                # --- SQL DATABASES (.db or .sqlite files) ---
                elif file_extension in ['.db', '.sqlite']:
                    # SQLite needs a physical file, so we trick it by creating a temporary one
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name
                        
                    conn = sqlite3.connect(tmp_path)
                    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
                    
                    st.success(f"Connected to SQL Database: {file.name}")
                    st.write("Tables found in this database:", tables['name'].tolist())
                    
                    # Auto-load the first table to preview it
                    if not tables.empty:
                        first_table = tables['name'].iloc[0]
                        df = pd.read_sql_query(f"SELECT * FROM {first_table}", conn)
                        st.session_state['datasets'][f"{file.name}_{first_table}"] = df
                        st.write(f"Preview of table `{first_table}`:")
                        st.dataframe(df.head())
                    conn.close()

                # --- UNSTRUCTURED DATA (Documents) ---
                elif file_extension == '.txt':
                    text_data = file.getvalue().decode("utf-8")
                    st.session_state['documents'][file.name] = text_data
                    st.success(f"Loaded Text Document: {file.name}")
                    st.text_area("Preview", text_data[:500] + "...", height=150)
                    
                elif file_extension == '.pdf':
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_data = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
                    st.session_state['documents'][file.name] = text_data
                    st.success(f"Loaded PDF: {file.name}")
                    st.text_area("Preview", text_data[:500] + "...", height=150)
                    
                elif file_extension == '.docx':
                    doc = docx.Document(file)
                    text_data = "\n".join([para.text for para in doc.paragraphs])
                    st.session_state['documents'][file.name] = text_data
                    st.success(f"Loaded Word Doc: {file.name}")
                    st.text_area("Preview", text_data[:500] + "...", height=150)

            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")
