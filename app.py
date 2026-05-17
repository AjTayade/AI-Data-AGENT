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
import io
import time
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import google.generativeai as genai
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

st.set_page_config(page_title="AI Data Agent SaaS", layout="wide")

# ──────────────────────────────────────────────────────────────
# 1. SETUP THE BRAIN & THE VAULT
# ──────────────────────────────────────────────────────────────
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-3-flash-preview')
    supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

    if 'session' in st.session_state and st.session_state['session'] is not None:
        try:
            supabase.postgrest.auth(st.session_state['session'].access_token)
            supabase.auth.set_session(
                st.session_state['session'].access_token,
                st.session_state['session'].refresh_token
            )
        except Exception:
            st.session_state['user'] = None
            st.session_state['session'] = None
except Exception as e:
    st.error(f"Setup Error: Please check your Streamlit Secrets. ({e})")
    st.stop()

# ──────────────────────────────────────────────────────────────
# 2. THE BOUNCER (LOGIN / REGISTER)
# ──────────────────────────────────────────────────────────────
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
            log_pass  = st.text_input("Password", type="password", key="log_pass")
            if st.button("Login", use_container_width=True, type="primary"):
                try:
                    response = supabase.auth.sign_in_with_password({"email": log_email, "password": log_pass})
                    st.session_state['user']    = response.user
                    st.session_state['session'] = response.session
                    st.rerun()
                except Exception:
                    st.error("Login failed. Please check your credentials.")

        with auth_tab2:
            reg_name  = st.text_input("First Name", key="reg_name")
            reg_email = st.text_input("New Email", key="reg_email")
            reg_pass  = st.text_input("New Password", type="password", key="reg_pass")
            if st.button("Register", use_container_width=True):
                try:
                    supabase.auth.sign_up({
                        "email": reg_email, "password": reg_pass,
                        "options": {"data": {"first_name": reg_name}}
                    })
                    st.success("Registration successful! You can now log in.")
                except Exception as e:
                    st.error(f"Registration failed: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────
# 2.5  SESSION RESTORE  (runs once per login / after any refresh)
# ──────────────────────────────────────────────────────────────
_uid = st.session_state['user'].id

if not st.session_state.get('_session_restored'):
    try:
        _raw_nb   = supabase.table("raw_datasets").select("notebook_name").eq("user_id", _uid).execute()
        _clean_nb = supabase.table("cleaned_datasets").select("notebook_name").eq("user_id", _uid).execute()
        _dash_nb  = supabase.table("saved_dashboards").select("notebook_name").eq("user_id", _uid).execute()
        
        _all_nb   = sorted(set(
            [r['notebook_name'] for r in _raw_nb.data] +
            [r['notebook_name'] for r in _clean_nb.data] +
            [r['notebook_name'] for r in _dash_nb.data]
        ))
        if _all_nb and not st.session_state.get('notebook_list'):
            st.session_state['notebook_list'] = _all_nb
        if st.session_state.get('notebook_list') and st.session_state.get('active_notebook') is None:
            st.session_state['active_notebook'] = st.session_state['notebook_list'][0]
        st.session_state['_session_restored'] = True
    except Exception:
        st.session_state['_session_restored'] = True

# ──────────────────────────────────────────────────────────────
# 3. THE SIDEBAR (PROFILE & NOTEBOOKS)
# ──────────────────────────────────────────────────────────────
user_name = st.session_state['user'].user_metadata.get('first_name', 'User')

st.sidebar.markdown("### 👤 My Profile")
st.sidebar.markdown(f"**Name:** {user_name}")
st.sidebar.markdown(f"**Email:** {st.session_state['user'].email}")

if st.sidebar.button("🚪 Log Out", use_container_width=True):
    for _k in ['user', 'session', 'notebook_list', 'active_notebook',
               'raw_cloud', 'clean_cloud', 'dashboards', 'messages', '_session_restored', '_loaded_nb']:
        st.session_state.pop(_k, None)
    supabase.auth.sign_out()
    st.rerun()

st.sidebar.divider()

if 'notebook_list'  not in st.session_state: st.session_state['notebook_list']  = []
if 'active_notebook' not in st.session_state: st.session_state['active_notebook'] = None
if 'show_nb_modal'  not in st.session_state: st.session_state['show_nb_modal']  = False
if 'confirm_del_nb' not in st.session_state: st.session_state['confirm_del_nb'] = False
if 'dashboards' not in st.session_state: st.session_state['dashboards'] = {}

st.sidebar.markdown("### 📚 My Notebooks")

# ── New Notebook Button ────────────────────────────────────────
if st.sidebar.button("📓 New Notebook", use_container_width=True):
    st.session_state['show_nb_modal']  = True
    st.session_state['confirm_del_nb'] = False

# ── New Notebook Modal ─────────────────────────────────────────
@st.dialog("Create a New Notebook")
def new_notebook_modal():
    nb_input = st.text_input("Notebook name", placeholder="e.g., Q3 Financials")
    col_ok, col_cancel = st.columns(2)
    with col_ok:
        if st.button("✅ Create", use_container_width=True, type="primary"):
            if nb_input and nb_input not in st.session_state['notebook_list']:
                st.session_state['notebook_list'].append(nb_input)
                st.session_state['active_notebook'] = nb_input
                st.session_state['show_nb_modal']   = False
                st.rerun()
            elif nb_input in st.session_state['notebook_list']:
                st.warning("A notebook with this name already exists.")
            else:
                st.warning("Please enter a name.")
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.session_state['show_nb_modal'] = False
            st.rerun()

if st.session_state['show_nb_modal']:
    new_notebook_modal()

# ── Notebook List: select button + 🗑 delete icon per row ─────
if st.session_state['notebook_list']:
    st.sidebar.divider()
    st.sidebar.markdown("**My Notebooks:**")

    for nb in st.session_state['notebook_list']:
        is_active  = (nb == st.session_state.get('active_notebook'))
        col_nm, col_dl = st.sidebar.columns([5, 1])
        with col_nm:
            label = f"**› {nb}**" if is_active else nb
            if st.button(label, key=f"nb_select_{nb}", use_container_width=True):
                st.session_state['active_notebook'] = nb
                st.session_state['confirm_del_nb']  = False
                st.rerun()
        with col_dl:
            if st.button("🗑", key=f"nb_del_{nb}", help=f"Delete notebook '{nb}'"):
                st.session_state['confirm_del_nb'] = True
                st.session_state['nb_to_delete']   = nb
                st.rerun()

# ── Delete Notebook Confirmation Dialog ───────────────────────
@st.dialog("⚠️ Confirm Notebook Deletion")
def confirm_delete_notebook():
    nb_del = st.session_state.get('nb_to_delete', '')
    st.warning(
        f"Are you sure you want to permanently delete **'{nb_del}'**?\n\n"
        f"This will remove **all raw, cleaned datasets, and dashboards** stored in this notebook. "
        f"This action **cannot be undone**."
    )
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("🗑️ Yes, Delete", use_container_width=True, type="primary"):
            try:
                supabase.table("raw_datasets").delete().eq("user_id", _uid).eq("notebook_name", nb_del).execute()
                supabase.table("cleaned_datasets").delete().eq("user_id", _uid).eq("notebook_name", nb_del).execute()
                supabase.table("saved_dashboards").delete().eq("user_id", _uid).eq("notebook_name", nb_del).execute()

                st.session_state['notebook_list'].remove(nb_del)
                remaining = st.session_state['notebook_list']
                st.session_state['active_notebook'] = remaining[0] if remaining else None
                for _k in ['raw_cloud', 'clean_cloud', '_loaded_nb']:
                    st.session_state.pop(_k, None)
                st.session_state['dashboards'].pop(nb_del, None)
                st.session_state['confirm_del_nb'] = False
                st.session_state.pop('nb_to_delete', None)
                st.rerun()
            except Exception as e:
                st.error(f"Could not delete notebook: {e}")
    with col_no:
        if st.button("Cancel", use_container_width=True):
            st.session_state['confirm_del_nb'] = False
            st.session_state.pop('nb_to_delete', None)
            st.rerun()

if st.session_state.get('confirm_del_nb'):
    confirm_delete_notebook()

# ──────────────────────────────────────────────────────────────
# 4. THE NOTEBOOK ENFORCER
# ──────────────────────────────────────────────────────────────
if st.session_state['active_notebook'] is None:
    st.warning("👋 Welcome! Please create your first Notebook in the sidebar to begin working.")
    st.stop()

st.title(f"📂 Notebook: {st.session_state['active_notebook']}")
st.divider()

# ──────────────────────────────────────────────────────────────
# 5. SECURE CLOUD FETCHING (only re-fetches when notebook changes)
# ──────────────────────────────────────────────────────────────
user_id = st.session_state['user'].id
nb_name = st.session_state['active_notebook']

_nb_changed = (st.session_state.get('_loaded_nb') != nb_name)

if _nb_changed or 'raw_cloud' not in st.session_state:
    st.session_state['raw_cloud']   = {}
    st.session_state['clean_cloud'] = {}
    st.session_state['dashboards']  = {} 
    with st.spinner(f"Loading {nb_name} vault..."):
        try:
            raw_res = supabase.table("raw_datasets").select("file_name, data") \
                .eq("user_id", user_id).eq("notebook_name", nb_name).execute()
            st.session_state['raw_cloud'] = {
                r['file_name']: pd.DataFrame(r['data']) for r in raw_res.data
            }
            clean_res = supabase.table("cleaned_datasets").select("file_name, data") \
                .eq("user_id", user_id).eq("notebook_name", nb_name).execute()
            st.session_state['clean_cloud'] = {
                r['file_name']: pd.DataFrame(r['data']) for r in clean_res.data
            }
            
            # Fetch saved dashboards and inflate from JSON back to Plotly objects
            dash_res = supabase.table("saved_dashboards").select("dashboard_data") \
                .eq("user_id", user_id).eq("notebook_name", nb_name).execute()
            if dash_res.data:
                saved_json = dash_res.data[0]['dashboard_data']
                st.session_state['dashboards'][nb_name] = {
                    title: pio.from_json(fig_json) for title, fig_json in saved_json.items()
                }
                
            st.session_state['_loaded_nb'] = nb_name
        except Exception as e:
            st.warning(f"Could not fetch cloud data: {e}")

# ──────────────────────────────────────────────────────────────
# TABBED ARCHITECTURE
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📤 Part 1 — Upload",
    "🧹 Part 2 — Sanitise & Query",
    "📊 Part 3 — Dashboard"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📤 Upload New Data")
    st.caption(
        "Supported: CSV, Excel (.xlsx/.xls), JSON, SQLite (.db/.sqlite), "
        "PDF, Word (.docx), plain text (.txt). Multiple files at once supported."
    )

    uploaded_files = st.file_uploader(
        f"Drop files into '{nb_name}'",
        type=["csv", "xlsx", "xls", "json", "db", "sqlite", "pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("🚀 Upload File(s) to Database", type="primary"):
        if not uploaded_files:
            st.warning("Please select at least one file to upload.")
        else:
            for file in uploaded_files:

                # ── GUARD 1: Already in this notebook ─────────────────
                if file.name in st.session_state.get('raw_cloud', {}):
                    st.warning(
                        f"⚠️ **'{file.name}'** already exists in this notebook. "
                        f"Skipped. To replace it, delete it in the file manager below."
                    )
                    continue

                # ── GUARD 2: Exists in another notebook — reuse ───────
                try:
                    existing_res = supabase.table("raw_datasets") \
                        .select("file_name, data, notebook_name") \
                        .eq("user_id", user_id).eq("file_name", file.name) \
                        .neq("notebook_name", nb_name).limit(1).execute()
                    if existing_res.data:
                        source_nb = existing_res.data[0]['notebook_name']
                        reused_df = pd.DataFrame(existing_res.data[0]['data']).replace({np.nan: None})
                        supabase.table('raw_datasets').insert({
                            'file_name': file.name, 'data': reused_df.to_dict(orient='records'),
                            'user_id': user_id, 'notebook_name': nb_name
                        }).execute()
                        st.session_state['raw_cloud'][file.name] = reused_df
                        st.info(f"♻️ **'{file.name}'** found in notebook **'{source_nb}'** — linked here.")
                        continue
                except Exception:
                    pass  # fall through to normal upload

                # ── NORMAL PARSE + UPLOAD ──────────────────────────────
                file_ext = pathlib.Path(file.name).suffix.lower()
                try:
                    if file_ext in ['.csv', '.xlsx', '.xls', '.json']:
                        if file_ext == '.csv':
                            df = pd.read_csv(file)
                        elif file_ext in ['.xlsx', '.xls']:
                            df = pd.read_excel(file)
                        elif file_ext == '.json':
                            try:
                                df = pd.read_json(file, orient='records')
                            except Exception:
                                file.seek(0)
                                try:
                                    df = pd.read_json(file, lines=True)
                                except Exception:
                                    file.seek(0)
                                    content = file.getvalue().decode("utf-8")
                                    objs = []
                                    for line in content.splitlines():
                                        if line.strip():
                                            try: objs.append(json.loads(line))
                                            except json.JSONDecodeError: pass
                                    if objs: df = pd.json_normalize(objs)
                                    else: raise Exception("Could not extract tabular JSON data.")
                        df = df.replace({np.nan: None})
                        with st.spinner(f"Saving {file.name}..."):
                            supabase.table('raw_datasets').insert({
                                'file_name': file.name, 'data': df.to_dict(orient='records'),
                                'user_id': user_id, 'notebook_name': nb_name
                            }).execute()
                        st.session_state['raw_cloud'][file.name] = df
                        st.success(f"✅ Uploaded: {file.name}")

                    elif file_ext in ['.txt', '.pdf', '.docx', '.db', '.sqlite']:
                        if file_ext == '.txt':
                            text_data = file.getvalue().decode("utf-8")
                        elif file_ext == '.pdf':
                            pdf_reader = PyPDF2.PdfReader(file)
                            text_data  = "".join([p.extract_text() + "\n" for p in pdf_reader.pages])
                        elif file_ext == '.docx':
                            doc_obj   = docx.Document(file)
                            text_data = "\n".join([p.text for p in doc_obj.paragraphs])
                        elif file_ext in ['.db', '.sqlite']:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                                tmp.write(file.read()); tmp_path = tmp.name
                            conn = sqlite3.connect(tmp_path)
                            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
                            text_data = f"SQLite DB: {file.name}\nTables: {tables['name'].tolist()}"
                            conn.close()
                        df = pd.DataFrame([{"document_name": file.name, "content": text_data}])
                        with st.spinner(f"Saving {file.name}..."):
                            supabase.table('raw_datasets').insert({
                                'file_name': file.name, 'data': df.to_dict(orient='records'),
                                'user_id': user_id, 'notebook_name': nb_name
                            }).execute()
                        st.session_state['raw_cloud'][file.name] = df
                        st.success(f"✅ Uploaded document: {file.name}")

                except Exception as e:
                    st.error(f"Error uploading {file.name}: {e}")

    # ── Success nudge ──────────────────────────────────────────
    if st.session_state.get('raw_cloud'):
        st.success(
            f"✅ {len(st.session_state['raw_cloud'])} file(s) ready — "
            f"head to **Part 2** to sanitise or **Part 3** for your dashboard."
        )

    # ── File Previews ──────────────────────────────────────────
    if st.session_state.get('raw_cloud'):
        st.divider()
        st.subheader("🗃️ Uploaded Files Preview")
        for fname, fdf in st.session_state['raw_cloud'].items():
            with st.expander(f"📄 {fname}  —  {len(fdf):,} rows × {len(fdf.columns)} cols"):
                st.dataframe(fdf.head(5), use_container_width=True)

    # ══ FILE MANAGER — Checkbox-based multi-delete ════════════
    st.divider()
    st.subheader("🗂️ Manage Files")

    all_stored_files = (
        [("raw_datasets",     k, "🟡 RAW")   for k in st.session_state.get('raw_cloud',   {}).keys()] +
        [("cleaned_datasets", k, "🟢 CLEAN") for k in st.session_state.get('clean_cloud', {}).keys()]
    )

    if not all_stored_files:
        st.caption("No files in this notebook yet.")
    else:
        # Select All toggle
        select_all = st.checkbox("☑️ Select All", key="del_select_all")

        checked_files = []
        for table_name, fname, badge in all_stored_files:
            col_chk, col_label = st.columns([1, 11])
            with col_chk:
                checked = st.checkbox(
                    "", key=f"del_chk_{table_name}_{fname}",
                    value=select_all, label_visibility="collapsed"
                )
            with col_label:
                st.markdown(f"{badge} &nbsp; `{fname}`", unsafe_allow_html=True)
            if checked:
                checked_files.append((table_name, fname))

        st.write("")
        if checked_files:
            st.caption(f"**{len(checked_files)}** file(s) selected for deletion")
            if st.button(
                f"🗑️ Delete {len(checked_files)} Selected File(s)",
                type="secondary", use_container_width=True
            ):
                errors = []
                for table_name, original_name in checked_files:
                    try:
                        supabase.table(table_name).delete() \
                            .eq("user_id",       user_id) \
                            .eq("notebook_name", nb_name) \
                            .eq("file_name",     original_name) \
                            .execute()
                        if table_name == "raw_datasets":
                            st.session_state['raw_cloud'].pop(original_name, None)
                        else:
                            st.session_state['clean_cloud'].pop(original_name, None)
                    except Exception as e:
                        errors.append(f"{original_name}: {e}")

                if errors:
                    for err in errors: st.error(f"🚨 {err}")
                else:
                    st.success(f"✅ {len(checked_files)} file(s) permanently deleted.")
                st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 2 — SANITISE & CHAT
# ══════════════════════════════════════════════════════════════
with tab2:

    if not st.session_state.get('raw_cloud'):
        st.info("⬆️ No data yet — upload your files in **Part 1** first.")

    # ── Sanitizer — sequential multi-file loop ─────────────────
    st.header("🧠 Universal Data Sanitizer")

    raw_keys = list(st.session_state.get('raw_cloud', {}).keys())
    if raw_keys:
        selected_files = st.multiselect(
            "Select raw dataset(s) to sanitise — each file processed in sequence:",
            raw_keys,
            default=raw_keys[:1]
        )

        if st.button("🪄 Run Exhaustive AI Sanitization", type="primary"):
            if not selected_files:
                st.warning("Please select at least one file to sanitize.")
            else:
                total_files  = len(selected_files)
                progress_bar = st.progress(0)
                status_text  = st.empty()

                for i, file in enumerate(selected_files):
                    est_time = (total_files - i) * 15
                    status_text.info(
                        f"Processing **{i+1} of {total_files}**: `{file}` "
                        f"— Est. remaining: ~{est_time}s"
                    )

                    df_to_process = st.session_state['raw_cloud'][file]
                    original_rows = len(df_to_process)
                    original_cols = len(df_to_process.columns)

                    # ── Rich column profile for the AI ────────
                    profile_lines = []
                    for col in df_to_process.columns:
                        dtype       = str(df_to_process[col].dtype)
                        null_pct    = round(df_to_process[col].isnull().mean() * 100, 1)
                        sample_vals = df_to_process[col].dropna().astype(str).unique()[:5].tolist()
                        profile_lines.append(
                            f"  - {col} | dtype={dtype} | nulls={null_pct}% | samples={sample_vals}"
                        )
                    col_profile = "\n".join(profile_lines)
                    full_sample = df_to_process.head(20).to_csv(index=False)

                    clean_prompt = f"""
You are a Senior Data Engineer and experienced Data Analyst. Write a single Python function
called `clean_data(df)` that exhaustively sanitizes the provided dataset for analysis.

DATASET PROFILE:
{col_profile}

SAMPLE DATA:
{full_sample}

YOUR FUNCTION MUST HANDLE ALL OF THE FOLLOWING — apply only what is relevant to this data:
1. Convert all column headers to strict snake_case.
2. Drop columns that are 100% empty (all NaN/None).
3. Drop rows that are 100% empty.
4. Drop exact duplicate rows. Reset the DataFrame index after dropping.
5. For every object/string column: strip leading/trailing whitespace via df[col].str.strip().
6. Remove hidden chars via df[col].str.replace() with a compiled regex per column.
7. Normalize multiple consecutive internal spaces to single space.
8. Replace junk null strings with np.nan (case-insensitive): '', 'n/a', 'na', 'none', 'null', 'nan', '-', '?', '#n/a'
9. Numeric columns: null rate <20% impute median; 20-50% leave NaN; >50% drop column.
10. Strip currency symbols ($,€,£,¥), comma separators, % from numeric strings and cast to float.
11. Object columns where all non-null values are numeric strings cast to float.
12. Outlier capping (Winsorize): for numeric cols where outlier count >1%, cap at Q1-1.5*IQR and Q3+1.5*IQR.
13. Detect date cols by keywords: 'date','time','dt','year','month','created'. Format ISO 8601: df[col].dt.strftime('%Y-%m-%d').
14. Casing: names/cities -> .str.title() | codes/IDs -> .str.upper() | text -> .str.lower()
15. Low-cardinality cols (<30 unique): map near-duplicate variants to most-frequent form.
16. Cols with values only from {{yes,no,true,false,1,0,y,n,t,f}} map to Python bool True/False.
17. ID/code/zip/phone cols: keep as string, no numeric cast, no leading-zero stripping.
18. if df.empty: raise ValueError("Cleaning resulted in empty DataFrame")

STRICT RULES:
- Always check if df[col].dtype == object before applying .str methods.
- Use df[col].str methods — NEVER df.applymap() or df.map() on whole DataFrame.
- Do NOT aggregate, pivot, merge, or reshape.
- Imports inside function body: import pandas as pd, import numpy as np, import re.
- Return ONLY valid executable Python. NO markdown. NO backticks. NO explanations.
"""

                    try:
                        clean_response = model.generate_content(clean_prompt)
                        clean_code = (
                            clean_response.text.strip()
                            .replace("```python", "").replace("```", "").strip()
                        )
                        local_vars = {}
                        exec(clean_code, globals(), local_vars)
                        df_cleaned = local_vars['clean_data'](df_to_process)

                        # ── Before / After ────────────────────
                        st.subheader(f"Results for: `{file}`")
                        b_col, a_col = st.columns(2)
                        with b_col:
                            st.markdown("**🔴 Before (Raw)**")
                            st.dataframe(df_to_process.head(5), use_container_width=True)
                            st.caption(f"{original_rows:,} rows · {original_cols} cols")
                        with a_col:
                            st.markdown("**🟢 After (Sanitised)**")
                            st.dataframe(df_cleaned.head(5), use_container_width=True)
                            st.caption(f"{len(df_cleaned):,} rows · {len(df_cleaned.columns)} cols")

                        # ── Auto-save ─────────────────────────
                        clean_name      = f"sanitized_{file}"
                        df_cleaned_json = df_cleaned.replace({np.nan: None})
                        supabase.table('cleaned_datasets').insert({
                            'file_name': clean_name,
                            'data':      df_cleaned_json.to_dict(orient='records'),
                            'user_id':   user_id, 'notebook_name': nb_name
                        }).execute()
                        st.session_state['clean_cloud'][clean_name] = df_cleaned
                        st.success(f"✅ Auto-saved as **'{clean_name}'**.")

                        # ── Format-aware download ─────────────
                        ext_dl = pathlib.Path(file).suffix.lower()
                        try:
                            if ext_dl == '.csv':
                                fb = df_cleaned.to_csv(index=False).encode('utf-8')
                                mime = "text/csv"; dl_name = clean_name
                            elif ext_dl in ['.xlsx', '.xls']:
                                buf = io.BytesIO()
                                with pd.ExcelWriter(buf, engine='openpyxl') as w:
                                    df_cleaned.to_excel(w, index=False)
                                fb = buf.getvalue()
                                mime = ("application/vnd.openxmlformats-officedocument"
                                        ".spreadsheetml.sheet")
                                dl_name = f"sanitized_{pathlib.Path(file).stem}.xlsx"
                            elif ext_dl == '.json':
                                fb = df_cleaned.to_json(orient='records', indent=2).encode('utf-8')
                                mime = "application/json"; dl_name = clean_name
                            else:
                                fb = df_cleaned.to_csv(index=False).encode('utf-8')
                                mime = "text/csv"
                                dl_name = f"sanitized_{pathlib.Path(file).stem}.csv"
                        except ImportError:
                            fb = df_cleaned.to_csv(index=False).encode('utf-8')
                            mime = "text/csv"
                            dl_name = f"sanitized_{pathlib.Path(file).stem}.csv"

                        st.download_button(
                            label=f"⬇️ Download {dl_name}",
                            data=fb, file_name=dl_name, mime=mime,
                            use_container_width=True
                        )
                        st.divider()

                    except Exception as e:
                        st.error(f"🚨 Sanitization failed for '{file}': {e}")

                    progress_bar.progress((i + 1) / total_files)

                status_text.success(f"🎉 All {total_files} file(s) sanitised and saved!")

    # ── Chat Section ───────────────────────────────────────────
    st.divider()
    st.header("💬 Chat with your Data")

    all_chat_data = {
        **st.session_state.get('raw_cloud',   {}),
        **st.session_state.get('clean_cloud', {})
    }

    if not all_chat_data:
        st.info("Upload or sanitise data above to begin chatting.")
    else:
        chat_file = st.selectbox(
            "Which dataset do you want to talk to?",
            list(all_chat_data.keys()),
            key="chat_file_selector"
        )
        df_chat = all_chat_data[chat_file]

        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", temperature=0,
            google_api_key=st.secrets["GEMINI_API_KEY"]
        )
        agent = create_pandas_dataframe_agent(
            llm, df_chat, verbose=True,
            allow_dangerous_code=True, handle_parsing_errors=True
        )

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
                        answer   = response["output"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_str = str(e)
                        if "Could not parse LLM output:" in error_str:
                            extracted = (
                                error_str
                                .split("Could not parse LLM output:")[1]
                                .split("For troubleshooting, visit:")[0]
                                .strip(" `\n")
                            )
                            st.markdown(extracted)
                            st.session_state.messages.append({"role": "assistant", "content": extracted})
                        else:
                            st.error(f"Could not calculate that. Error: {e}")

# ══════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab3:
    all_dash_data = {
        **st.session_state.get('raw_cloud',   {}),
        **st.session_state.get('clean_cloud', {})
    }

    if not all_dash_data:
        st.info("⬆️ No data yet — upload your files in **Part 1** first.")
    else:
        st.header("📊 Executive Analytics")

        # ── Check if dashboard already exists in memory ──
        if nb_name in st.session_state.get('dashboards', {}):
            st.success("✅ Persistent Dashboard Loaded from Vault.")
            current_dash = st.session_state['dashboards'][nb_name]
            
            # --- EXPORT TO HTML ---
            with st.expander("📥 Export Dashboard Report", expanded=False):
                st.caption("Download this dashboard as a standalone, interactive HTML file to share with stakeholders.")
                html_content = f"<html><head><title>Executive Report: {nb_name}</title><style>body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background-color: #f4f6f9; padding: 40px; color: #1a1a2e; }} .container {{ max-width: 1200px; margin: auto; }} h1 {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; color: #2c3e50; }} .chart-box {{ background: white; margin-bottom: 30px; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}</style></head><body><div class='container'><h1>Executive Report: {nb_name}</h1>"
                for title, fig in current_dash.items():
                    html_content += f"<div class='chart-box'><h2>{title}</h2>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
                html_content += "</div></body></html>"

                st.download_button(label="📄 Download Interactive HTML Report", data=html_content, file_name=f"Executive_Report_{nb_name}.html", mime="text/html", use_container_width=True)
            
            # --- RENDER CHARTS ---
            chart_titles = list(current_dash.keys())
            for row_start in range(0, len(chart_titles), 2):
                c1, c2 = st.columns(2)
                with c1:
                    t = chart_titles[row_start]
                    st.plotly_chart(current_dash[t], use_container_width=True)
                if row_start + 1 < len(chart_titles):
                    with c2:
                        t = chart_titles[row_start + 1]
                        st.plotly_chart(current_dash[t], use_container_width=True)
            
            # Allow user to overwrite
            st.divider()
            if st.button("🔄 Regenerate New Dashboard Configuration", type="secondary"):
                supabase.table("saved_dashboards").delete().eq("user_id", user_id).eq("notebook_name", nb_name).execute()
                del st.session_state['dashboards'][nb_name]
                st.rerun()

        # ── Dashboard Builder UI (If none exists) ──
        else:
            st.caption("Select datasets to analyze. The AI will detect patterns and generate a story-driven, presentation-ready dashboard.")
            clean_keys   = list(st.session_state.get('clean_cloud', {}).keys())
            default_keys = clean_keys[:2] if clean_keys else list(all_dash_data.keys())[:1]

            dash_files = st.multiselect("Select dataset(s) to include:", list(all_dash_data.keys()), default=default_keys)

            if dash_files:
                df_dict = {f: all_dash_data[f] for f in dash_files}
                
                schema_intel = ""
                for name, df in df_dict.items():
                    schema_intel += f"\n=== {name} ===\nCols: {list(df.columns)}\n"

                if st.button("🪄 Generate Executive Dashboard", type="primary"):
                    with st.spinner("AI is analyzing schemas and designing visuals..."):
                        dash_prompt = f"""
You are a Senior BI Architect designing a dashboard for a Fortune 500 executive.
Write a Python function `build_dashboard(data_dict)` returning EXACTLY 6 Plotly figures.

DATASETS:
{schema_intel}

EXECUTIVE DESIGN RULES (STRICT):
1. TITLES TELL A STORY: Never use "Sales vs Time". Use "Sales Peaked in Q4 Due to X". Infer likely trends from column names.
2. MINIMALIST AESTHETIC: You MUST apply `template="plotly_white"` to every chart.
3. REMOVE JUNK: Hide gridlines `fig.update_yaxes(showgrid=False)`. Hide top/right spines.
4. SOPHISTICATED COLORS: Use professional palettes like `px.colors.qualitative.Prism`.
5. ANNOTATIONS: Add `text_auto='.2s'` to bar charts. 
6. MIX: 1 Trend (Line), 1 Distribution (Hist/Box), 1 Comparison (Bar), 1 Relationship (Scatter), 2 others.

FUNCTION CONTRACT:
- Extract: `df = data_dict["key"]`. Dynamically infer dtypes `df.select_dtypes(...)`.
- Handle NaNs: `df_plot = df.dropna(...)`.
- Wrap EVERY chart in `try/except`. If one fails, generate a fallback text-based figure using `go.Figure()`.
- Return exactly: `{{ "Story Title 1": fig1, "Story Title 2": fig2, "Story Title 3": fig3, "Story Title 4": fig4, "Story Title 5": fig5, "Story Title 6": fig6 }}`
- Return ONLY valid Python code. NO backticks. NO markdown.
"""
                        try:
                            dash_response = model.generate_content(dash_prompt)
                            dash_code = dash_response.text.strip().replace("```python", "").replace("```", "").strip()
                            
                            local_vars = {}
                            safe_globals = globals().copy()
                            safe_globals['px'] = px
                            safe_globals['go'] = go
                            safe_globals['pd'] = pd

                            exec(dash_code, safe_globals, local_vars)
                            dashboard_figs = local_vars['build_dashboard'](df_dict)

                            # --- SAVING LOGIC ---
                            # 1. Convert complex Plotly objects to JSON strings
                            dash_json = {title: fig.to_json() for title, fig in dashboard_figs.items()}
                            
                            # 2. Save to Supabase (Upsert to replace if exists)
                            supabase.table('saved_dashboards').upsert({
                                'user_id': user_id,
                                'notebook_name': nb_name,
                                'dashboard_data': dash_json
                            }, on_conflict='user_id, notebook_name').execute()

                            # 3. Save to local memory and reload
                            st.session_state.setdefault('dashboards', {})[nb_name] = dashboard_figs
                            st.rerun()

                        except Exception as e:
                            st.error(f"🚨 Dashboard generation failed. Details: {e}")
