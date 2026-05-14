import streamlit as st
import pandas as pd
import pathlib
import sqlite3
import tempfile
import PyPDF2
import docx
import re             
import numpy as np
import google.generativeai as genai
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

st.set_page_config(page_title="AI Data Agent", layout="wide")


# --- 1. SETUP THE BRAIN & THE VAULT ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    # CORRECTED: Using an official Gemini 3 reasoning model ID
    model = genai.GenerativeModel('gemini-3-flash-preview')

    supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
except Exception as e:
    # This will now catch and display if the API key itself is missing or invalid
    st.error(f"Setup Error: Please check your Streamlit Secrets or Model ID. ({e})")
    st.stop()
    
# --- 2. FETCH CLOUD DATA ON STARTUP ---
if 'raw_cloud' not in st.session_state:
    st.session_state['raw_cloud'] = {}
    st.session_state['clean_cloud'] = {}
    with st.spinner("Syncing with Supabase Cloud..."):
        try:
            raw_res = supabase.table("raw_datasets").select("file_name, data").execute()
            st.session_state['raw_cloud'] = {r['file_name']: pd.DataFrame(r['data']) for r in raw_res.data}
            
            clean_res = supabase.table("cleaned_datasets").select("file_name, data").execute()
            st.session_state['clean_cloud'] = {r['file_name']: pd.DataFrame(r['data']) for r in clean_res.data}
        except Exception as e:
            st.warning(f"Could not fetch cloud data: {e}")

colA, colB = st.columns(2)
with colA:
    if st.session_state['raw_cloud']:
        with st.expander("☁️ View Raw Files in Database"):
            for name, df in st.session_state['raw_cloud'].items():
                st.write(f"**{name}**")
                st.dataframe(df.head(3), use_container_width=True)
with colB:
    if st.session_state['clean_cloud']:
        with st.expander("✨ View Cleaned Files in Database"):
            for name, df in st.session_state['clean_cloud'].items():
                st.write(f"**{name}**")
                st.dataframe(df.head(3), use_container_width=True)

# --- 3. THE UNIVERSAL AUTO-SAVE LOADER ---
st.divider()
st.subheader("📤 Upload New Data")
# RESTORED: All file types are back!
uploaded_files = st.file_uploader(
    "Drop your files here", 
    type=["csv", "xlsx", "xls", "json", "db", "sqlite", "pdf", "docx", "txt"], 
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        file_ext = pathlib.Path(file.name).suffix.lower()
        
        try:
           # -- STRUCTURED DATA --
            if file_ext in ['.csv', '.xlsx', '.xls', '.json']:
                if file.name not in st.session_state['raw_cloud']:
                    if file_ext == '.csv': df = pd.read_csv(file)
                    elif file_ext in ['.xlsx', '.xls']: df = pd.read_excel(file)
                    elif file_ext == '.json': df = pd.read_json(file, orient='records')
                    
                    # 🛠️ THE FIX: Convert all 'NaN' values to 'None' for JSON compliance
                    import numpy as np
                    df = df.replace({np.nan: None})
                    
                    with st.spinner(f"Auto-saving {file.name} to cloud..."):
                        supabase.table('raw_datasets').upsert({
                            'file_name': file.name, 
                            'data': df.to_dict(orient='records')
                        }).execute()
                    st.session_state['raw_cloud'][file.name] = df 
                    st.success(f"✅ Loaded and saved: {file.name}")
            
            # -- SQL DATABASES --
            elif file_ext in ['.db', '.sqlite']:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                    
                conn = sqlite3.connect(tmp_path)
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
                
                if not tables.empty:
                    for _, row in tables.iterrows():
                        table_name = row['name']
                        full_name = f"{file.name}_{table_name}" # e.g. mydata.db_users
                        if full_name not in st.session_state['raw_cloud']:
                            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                            with st.spinner(f"Auto-saving SQL table {full_name}..."):
                                supabase.table('raw_datasets').upsert({'file_name': full_name, 'data': df.to_dict(orient='records')}).execute()
                            st.session_state['raw_cloud'][full_name] = df
                    st.success(f"✅ Extracted and saved SQL Database: {file.name}")
                conn.close()

            # -- UNSTRUCTURED DOCUMENTS --
            elif file_ext in ['.txt', '.pdf', '.docx']:
                if file.name not in st.session_state['raw_cloud']:
                    text_data = ""
                    if file_ext == '.txt':
                        text_data = file.getvalue().decode("utf-8")
                    elif file_ext == '.pdf':
                        pdf_reader = PyPDF2.PdfReader(file)
                        text_data = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
                    elif file_ext == '.docx':
                        doc = docx.Document(file)
                        text_data = "\n".join([para.text for para in doc.paragraphs])
                        
                    # Trick the database: Pack text into a DataFrame!
                    df = pd.DataFrame([{"document_name": file.name, "content": text_data}])
                    
                    with st.spinner(f"Auto-saving Document {file.name}..."):
                        supabase.table('raw_datasets').upsert({'file_name': file.name, 'data': df.to_dict(orient='records')}).execute()
                    st.session_state['raw_cloud'][file.name] = df 
                    st.success(f"✅ Loaded and saved Document: {file.name}")

        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")

# --- 4. THE AI DOMAIN SPECIALIST ---
if st.session_state['raw_cloud']:
    st.divider()
    st.header("🧠 Step 3: AI Domain Specialist")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_file = st.selectbox("Select a raw dataset to process:", list(st.session_state['raw_cloud'].keys()))
    with col2:
        domain_context = st.text_input("Industry/Domain:", value="General Business")
    
    if st.button("🪄 Run AI Data Cleaning & Analysis", type="primary"):
        df_to_process = st.session_state['raw_cloud'][selected_file]
        
        with st.spinner(f"AI is sanitizing {selected_file}..."):
            # 1. The "Universal Sanitizer" Prompt
            clean_prompt = f"""
            You are a Senior Data Engineer. Your task is to write a Python function 'clean_data(df)' 
            to sanitize this dataset for any irregularities. 
            
            INPUT DATA INFO:
            Sample: {df_to_process.head(5).to_csv(index=False)}
            Schema: {str(df_to_process.dtypes)}

            THE FUNCTION MUST:
            1. Strip all leading/trailing whitespace from string columns.
            2. Detect and remove hidden non-printable characters (e.g., \\x00, \\ufeff).
            3. Standardize date columns to ISO format if detected.
            4. Identify 'junk' values (like '?', 'N/A', 'none', '---') and convert them to standard Nulls (np.nan).
            5. Drop rows that are 100% empty and drop columns that are 100% null.
            6. Handle inconsistent casing (e.g., convert all headers to snake_case).

            Return ONLY valid executable Python code. NO markdown. NO backticks. NO explanations.
            Include: import pandas as pd, import numpy as np.
            """

            try:
                # 2. Send to Gemini
                clean_response = model.generate_content(clean_prompt)
                clean_code = clean_response.text.strip().replace("```python", "").replace("```", "").strip()
                
                # 3. Execute the AI's code locally
                local_vars = {}
                exec(clean_code, globals(), local_vars)
                df_cleaned = local_vars['clean_data'](df_to_process)
                
                st.success("✅ Data sanitized for all irregularities!")
                
                # 4. Display results & Download/Save options
                clean_name = f"sanitized_{selected_file}"
                
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("⬇️ Download Sanitized CSV", df_cleaned.to_csv(index=False), f"{clean_name}.csv", "text/csv")
                with c2:
                    if st.button(f"💾 Save {clean_name} to Cloud"):
                        supabase.table('cleaned_datasets').upsert({'file_name': clean_name, 'data': df_cleaned.to_dict(orient='records')}).execute()
                        st.session_state['clean_cloud'][clean_name] = df_cleaned
                        st.success("Saved!")

            except Exception as e:
                st.error(f"🚨 Sanitization failed: {e}")

# --- 5. THE DATA WHISPERER (Step 4) ---
all_chat_data = {**st.session_state.get('raw_cloud', {}), **st.session_state.get('clean_cloud', {})}

if all_chat_data:
    st.divider()
    st.header("💬 Step 4: Chat with your Data")
    
    chat_file = st.selectbox("Which dataset do you want to talk to?", list(all_chat_data.keys()))
    df_chat = all_chat_data[chat_file]
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=st.secrets["GEMINI_API_KEY"])
    agent = create_pandas_dataframe_agent(llm, df_chat, verbose=True, allow_dangerous_code=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if prompt := st.chat_input(f"Ask something about {chat_file}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Crunching the numbers..."):
                try:
                    response = agent.invoke(prompt)
                    answer = response["output"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Could not calculate that. Error: {e}")
