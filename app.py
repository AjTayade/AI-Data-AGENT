import streamlit as st
import pandas as pd
import pathlib
import sqlite3
import tempfile
import PyPDF2
import docx
import re              
import numpy as np
import json
import google.generativeai as genai
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

st.set_page_config(page_title="AI Data Agent SaaS", layout="wide")

# --- 1. SETUP THE BRAIN & THE VAULT ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-3-flash-preview') 
    supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    
    if 'session' in st.session_state and st.session_state['session'] is not None:
        try:
            supabase.postgrest.auth(st.session_state['session'].access_token)
            supabase.auth.set_session(st.session_state['session'].access_token, st.session_state['session'].refresh_token)
        except Exception:
            st.session_state['user'] = None
            st.session_state['session'] = None
except Exception as e:
    st.error(f"Setup Error: Please check your Streamlit Secrets. ({e})")
    st.stop()

# --- 2. THE BOUNCER (LOGIN / REGISTER) ---
if 'user' not in st.session_state:
    st.session_state['user'] = None
if 'session' not in st.session_state:
    st.session_state['session'] = None

if st.session_state['user'] is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Welcome to AI Data Agent")
        st.markdown("Please log in to access your secure workspace.")
        
        auth_tab1, auth_tab2 = st.tabs(["Login", "Register"])
        with auth_tab1:
            log_email = st.text_input("Email", key="log_email")
            log_pass = st.text_input("Password", type="password", key="log_pass")
            if st.button("Login", use_container_width=True, type="primary"):
                try:
                    response = supabase.auth.sign_in_with_password({"email": log_email, "password": log_pass})
                    st.session_state['user'] = response.user
                    st.session_state['session'] = response.session 
                    st.rerun() 
                except Exception as e:
                    st.error("Login failed. Please check your credentials.")
                    
        with auth_tab2:
            reg_name = st.text_input("First Name", key="reg_name")
            reg_email = st.text_input("New Email", key="reg_email")
            reg_pass = st.text_input("New Password", type="password", key="reg_pass")
            if st.button("Register", use_container_width=True):
                try:
                    response = supabase.auth.sign_up({
                        "email": reg_email, "password": reg_pass, "options": {"data": {"first_name": reg_name}}
                    })
                    st.success("Registration successful! You can now log in.")
                except Exception as e:
                    st.error(f"Registration failed: {e}")
    st.stop() 

# --- 3. THE SIDEBAR (PROFILE & NOTEBOOKS) ---
user_name = st.session_state['user'].user_metadata.get('first_name', 'User')

st.sidebar.markdown("### 👤 My Profile")
st.sidebar.markdown(f"**Name:** {user_name}")
st.sidebar.markdown(f"**Email:** {st.session_state['user'].email}")

if st.sidebar.button("🚪 Log Out", use_container_width=True):
    st.session_state['user'] = None
    st.session_state['session'] = None
    supabase.auth.sign_out()
    st.rerun()

st.sidebar.divider()

if 'notebook_list' not in st.session_state:
    st.session_state['notebook_list'] = []

if 'active_notebook' not in st.session_state:
    st.session_state['active_notebook'] = None

st.sidebar.markdown("### 📚 My Notebooks")
new_nb_name = st.sidebar.text_input("Create New Notebook:", placeholder="e.g., Q3 Financials")

if st.sidebar.button("➕ Add Notebook", use_container_width=True):
    if new_nb_name and new_nb_name not in st.session_state['notebook_list']:
        st.session_state['notebook_list'].append(new_nb_name)
        st.session_state['active_notebook'] = new_nb_name
        st.rerun()

if st.session_state['notebook_list']:
    st.sidebar.divider()
    st.sidebar.markdown("**Switch Notebook:**")
    st.session_state['active_notebook'] = st.sidebar.radio(
        "Select active workspace:", st.session_state['notebook_list'], label_visibility="collapsed"
    )

# --- 4. THE NOTEBOOK ENFORCER ---
if st.session_state['active_notebook'] is None:
    st.warning("👋 Welcome! Please create your first Notebook in the sidebar to begin working.")
    st.stop()

st.title(f"📂 Notebook: {st.session_state['active_notebook']}")
st.divider()

# --- 5. SECURE CLOUD FETCHING ---
user_id = st.session_state['user'].id
nb_name = st.session_state['active_notebook']

st.session_state['raw_cloud'] = {}
st.session_state['clean_cloud'] = {}

with st.spinner(f"Loading {nb_name} secure vault..."):
    try:
        raw_res = supabase.table("raw_datasets").select("file_name, data").eq("user_id", user_id).eq("notebook_name", nb_name).execute()
        st.session_state['raw_cloud'] = {r['file_name']: pd.DataFrame(r['data']) for r in raw_res.data}
        
        clean_res = supabase.table("cleaned_datasets").select("file_name, data").eq("user_id", user_id).eq("notebook_name", nb_name).execute()
        st.session_state['clean_cloud'] = {r['file_name']: pd.DataFrame(r['data']) for r in clean_res.data}
    except Exception as e:
        st.warning(f"Could not fetch cloud data: {e}")

colA, colB = st.columns(2)
with colA:
    with st.expander("☁️ View Raw Files in Database", expanded=False):
        if st.session_state['raw_cloud']:
            for name, df in st.session_state['raw_cloud'].items():
                st.write(f"**{name}**")
                st.dataframe(df.head(3), use_container_width=True)
        else:
            st.caption("No raw files uploaded yet.")
with colB:
    with st.expander("✨ View Cleaned Files in Database", expanded=False):
        if st.session_state['clean_cloud']:
            for name, df in st.session_state['clean_cloud'].items():
                st.write(f"**{name}**")
                st.dataframe(df.head(3), use_container_width=True)
        else:
            st.caption("No cleaned files saved yet.")

# --- 6. SECURE AUTO-SAVE LOADER (ALWAYS VISIBLE) ---
st.divider()
st.subheader("📤 Step 1: Upload New Data")
uploaded_files = st.file_uploader(
    f"Drop files to save into '{nb_name}'", 
    type=["csv", "xlsx", "xls", "json", "db", "sqlite", "pdf", "docx", "txt"], 
    accept_multiple_files=True
)

if st.button(f"🚀 Upload File(s) to Database", type="primary"):
    if not uploaded_files:
        st.warning("Please select at least one file to upload.")
    else:
        for file in uploaded_files:
            file_ext = pathlib.Path(file.name).suffix.lower()
            try:
                if file_ext in ['.csv', '.xlsx', '.xls', '.json']:
                    if file_ext == '.csv': df = pd.read_csv(file)
                    elif file_ext in ['.xlsx', '.xls']: df = pd.read_excel(file)
                    elif file_ext == '.json': 
                        # ULTIMATE JSON FALLBACK PARSER
                        try:
                            df = pd.read_json(file, orient='records')
                        except Exception:
                            file.seek(0)
                            try:
                                df = pd.read_json(file, lines=True)
                            except Exception:
                                file.seek(0)
                                content = file.getvalue().decode("utf-8")
                                valid_json_objects = []
                                for line in content.splitlines():
                                    if line.strip():
                                        try:
                                            valid_json_objects.append(json.loads(line))
                                        except json.JSONDecodeError:
                                            pass
                                if valid_json_objects:
                                    df = pd.json_normalize(valid_json_objects)
                                else:
                                    raise Exception("Could not extract tabular JSON data.")
                    
                    df = df.replace({np.nan: None})
                    with st.spinner(f"Saving {file.name}..."):
                        # Insert handles the save without needing a unique constraint check
                        supabase.table('raw_datasets').insert({
                            'file_name': file.name, 'data': df.to_dict(orient='records'), 'user_id': user_id, 'notebook_name': nb_name
                        }).execute()
                    st.session_state['raw_cloud'][file.name] = df 
                    st.success(f"✅ Loaded and saved: {file.name}")
                
                elif file_ext in ['.txt', '.pdf', '.docx']:
                    text_data = ""
                    if file_ext == '.txt': text_data = file.getvalue().decode("utf-8")
                    elif file_ext == '.pdf':
                        pdf_reader = PyPDF2.PdfReader(file)
                        text_data = "".join([page.extract_text() + "\n" for page in pdf_reader.pages])
                    elif file_ext == '.docx':
                        doc = docx.Document(file)
                        text_data = "\n".join([para.text for para in doc.paragraphs])
                    
                    df = pd.DataFrame([{"document_name": file.name, "content": text_data}])
                    with st.spinner(f"Saving Document {file.name}..."):
                        supabase.table('raw_datasets').insert({
                            'file_name': file.name, 'data': df.to_dict(orient='records'), 'user_id': user_id, 'notebook_name': nb_name
                        }).execute()
                    st.session_state['raw_cloud'][file.name] = df 
                    st.success(f"✅ Loaded and saved Document: {file.name}")
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")

# --- 7. SECURE SANITIZER (ALWAYS VISIBLE) ---
st.divider()
st.header("🧠 Step 2: Universal Data Sanitizer")
if not st.session_state['raw_cloud']:
    st.info("No raw data available. Please upload a file above to begin sanitization.")
else:
    selected_file = st.selectbox("Select a raw dataset to process:", list(st.session_state['raw_cloud'].keys()))
    if st.button("🪄 Run Exhaustive AI Sanitization", type="primary"):
        df_to_process = st.session_state['raw_cloud'][selected_file]
        with st.spinner(f"AI is performing a deep clean on {selected_file}..."):
            clean_prompt = f"""
            You are a Senior Data Engineer. Write a Python function 'clean_data(df)' to exhaustively sanitize this dataset.
            Sample: {df_to_process.head(5).to_csv(index=False)}
            THE FUNCTION MUST:
            1. Strip whitespace from string columns.
            2. Remove hidden characters (e.g., \\x00, \\ufeff).
            3. Standardize dates to ISO format.
            4. Convert 'junk' values ('?', 'N/A') to standard np.nan.
            5. Drop 100% empty rows/columns.
            6. Convert headers to strict snake_case.
            7. Drop exact duplicate rows.
            8. IMPORTANT: Do NOT use `df.applymap()`. You MUST use `df.map()`.
            Return ONLY valid executable Python code. NO markdown. NO backticks.
            Include: import pandas as pd, import numpy as np, import re
            """
            try:
                clean_response = model.generate_content(clean_prompt)
                clean_code = clean_response.text.strip().replace("```python", "").replace("```", "").strip()
                local_vars = {}
                exec(clean_code, globals(), local_vars)
                df_cleaned = local_vars['clean_data'](df_to_process)
                
                st.success("✅ Deep Sanitization Complete!")
                st.subheader("🔍 Before & After Sanitization")
                b_col, a_col = st.columns(2)
                with b_col:
                    st.markdown("### 🔴 Before (Raw)")
                    st.dataframe(df_to_process.head(4), use_container_width=True)
                with a_col:
                    st.markdown("### 🟢 After (Sanitized)")
                    st.dataframe(df_cleaned.head(4), use_container_width=True)
                
                clean_name = f"sanitized_{selected_file}"
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("⬇️ Download Sanitized CSV", df_cleaned.to_csv(index=False).encode('utf-8'), f"{clean_name}.csv", "text/csv")
                with c2:
                    if st.button(f"💾 Save {clean_name} to Notebook"):
                        df_cleaned_json = df_cleaned.replace({np.nan: None})
                        supabase.table('cleaned_datasets').insert({
                            'file_name': clean_name, 'data': df_cleaned_json.to_dict(orient='records'), 'user_id': user_id, 'notebook_name': nb_name
                        }).execute()
                        st.session_state['clean_cloud'][clean_name] = df_cleaned
                        st.success("Saved to Cloud!")
            except Exception as e:
                st.error(f"🚨 Sanitization failed: {e}")

# --- 8. SECURE CHAT (ALWAYS VISIBLE) ---
all_chat_data = {**st.session_state.get('raw_cloud', {}), **st.session_state.get('clean_cloud', {})}

st.divider()
st.header("💬 Step 3: Chat with your Data")
if not all_chat_data:
    st.info("No data available to chat with. Please upload or sanitize a dataset first.")
else:
    chat_file = st.selectbox("Which dataset do you want to talk to?", list(all_chat_data.keys()))
    df_chat = all_chat_data[chat_file]
    
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0, google_api_key=st.secrets["GEMINI_API_KEY"])
    agent = create_pandas_dataframe_agent(llm, df_chat, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)
    
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
                    error_str = str(e)
                    if "Could not parse LLM output:" in error_str:
                        extracted = error_str.split("Could not parse LLM output:")[1].split("For troubleshooting, visit:")[0].strip(" `\n")
                        st.markdown(extracted)
                        st.session_state.messages.append({"role": "assistant", "content": extracted})
                    else:
                        st.error(f"Could not calculate that. Error: {e}")

# --- 9. SECURE DASHBOARD (ALWAYS VISIBLE) ---
st.divider()
st.header("📊 Step 4: Multi-File Dashboard")
if not all_chat_data:
    st.info("No data available for dashboards. Please upload data first.")
else:
    dash_files = st.multiselect("Select dataset(s) to include in the dashboard:", list(all_chat_data.keys()), default=list(all_chat_data.keys())[:1])
    
    if dash_files:
        df_dict = {f: all_chat_data[f] for f in dash_files}
        total_records = sum([len(df) for df in df_dict.values()])
        total_features = sum([len(df.columns) for df in df_dict.values()])
        
        st.subheader("Aggregated Data Scope")
        kpi1, kpi2 = st.columns(2)
        kpi1.metric("Total Records Across Files", f"{total_records:,}")
        kpi2.metric("Total Features Across Files", total_features)
        st.divider()
        
        if st.button("🪄 Auto-Generate Full Dashboard", type="primary"):
            with st.spinner("The AI is analyzing relationships across your files..."):
                schema_context = "\n".join([f"- {name}: {list(df.columns)}" for name, df in df_dict.items()])
                dash_prompt = f"""
                You are a Senior BI Developer. Write a Python function `build_dashboard(data_dict)` 
                that takes a dictionary of pandas DataFrames and generates 4 insightful Plotly charts.
                INPUT DATA INFO:
                {schema_context}
                THE FUNCTION MUST:
                1. Import plotly.express as px
                2. Extract dataframes from `data_dict` using the exact keys.
                3. Create 4 different charts (bar, scatter, pie, etc.).
                4. Return a dictionary: {{"Title 1": fig1, "Title 2": fig2, "Title 3": fig3, "Title 4": fig4}}
                Return ONLY valid executable Python code. NO markdown. NO backticks.
                """
                try:
                    dash_response = model.generate_content(dash_prompt)
                    dash_code = dash_response.text.strip().replace("```python", "").replace("```", "").strip()
                    local_vars = {}
                    exec(dash_code, globals(), local_vars)
                    dashboard_figs = local_vars['build_dashboard'](df_dict)
                    
                    st.success("✅ Dashboard Generated!")
                    chart_titles = list(dashboard_figs.keys())
                    
                    row1_col1, row1_col2 = st.columns(2)
                    with row1_col1:
                        st.subheader(chart_titles[0])
                        st.plotly_chart(dashboard_figs[chart_titles[0]], use_container_width=True)
                    with row1_col2:
                        st.subheader(chart_titles[1])
                        st.plotly_chart(dashboard_figs[chart_titles[1]], use_container_width=True)
                        
                    row2_col1, row2_col2 = st.columns(2)
                    with row2_col1:
                        st.subheader(chart_titles[2])
                        st.plotly_chart(dashboard_figs[chart_titles[2]], use_container_width=True)
                    with row2_col2:
                        st.subheader(chart_titles[3])
                        st.plotly_chart(dashboard_figs[chart_titles[3]], use_container_width=True)
                except Exception as e:
                    st.error(f"🚨 The AI struggled to build the dashboard. Error details: {e}")
