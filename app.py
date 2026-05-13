import streamlit as st
import pandas as pd
import pathlib
import google.generativeai as genai
from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

st.set_page_config(page_title="AI Data Agent", layout="wide")

# --- 1. SETUP THE BRAIN & THE VAULT ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
except Exception as e:
    st.error(f"Setup Error: Please check your Streamlit Secrets. ({e})")
    st.stop()

st.title("🗂️ The Ultimate Data Agent")

# --- 2. FETCH CLOUD DATA ON STARTUP ---
if 'raw_cloud' not in st.session_state:
    st.session_state['raw_cloud'] = {}
    st.session_state['clean_cloud'] = {}
    with st.spinner("Syncing with Supabase Cloud..."):
        try:
            # Fetch Raw
            raw_res = supabase.table("raw_datasets").select("file_name, data").execute()
            st.session_state['raw_cloud'] = {r['file_name']: pd.DataFrame(r['data']) for r in raw_res.data}
            # Fetch Cleaned
            clean_res = supabase.table("cleaned_datasets").select("file_name, data").execute()
            st.session_state['clean_cloud'] = {r['file_name']: pd.DataFrame(r['data']) for r in clean_res.data}
        except Exception as e:
            st.warning(f"Could not fetch cloud data: {e}")

# Display Cloud Memory
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

# --- 3. THE AUTO-SAVE LOADER ---
st.divider()
st.subheader("📤 Upload New Data")
uploaded_files = st.file_uploader("Drop your files here", type=["csv", "xlsx", "json"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state['raw_cloud']: # Only process if it's new
            file_ext = pathlib.Path(file.name).suffix.lower()
            try:
                if file_ext == '.csv': df = pd.read_csv(file)
                elif file_ext in ['.xlsx', '.xls']: df = pd.read_excel(file)
                elif file_ext == '.json': df = pd.read_json(file, orient='records')
                
                # Automatically save to Supabase Raw Table
                with st.spinner(f"Auto-saving {file.name} to cloud..."):
                    json_data = df.to_dict(orient='records')
                    supabase.table('raw_datasets').upsert({
                        'file_name': file.name,
                        'data': json_data
                    }).execute()
                
                st.session_state['raw_cloud'][file.name] = df # Update local memory
                st.success(f"✅ Loaded and saved to Raw Database: {file.name}")
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
        
        with st.spinner(f"AI is analyzing {selected_file}..."):
            clean_prompt = f"You are a Data Engineer in {domain_context}. Sample: {df_to_process.head(5).to_csv(index=False)}. Schema: {str(df_to_process.dtypes)}. Write a Python function 'clean_data(df)' to clean this. Return ONLY valid python code."
            
            try:
                clean_response = model.generate_content(clean_prompt)
                
                # ---> THE BUG WAS RIGHT HERE. IT IS NOW FIXED! <---
                clean_code = clean_response.text.strip().replace("```python", "").replace("
```", "").strip()
                
                local_vars = {}
                exec(clean_code, globals(), local_vars)
                df_cleaned = local_vars['clean_data'](df_to_process)
                st.success("✅ Data dynamically cleaned by AI!")
                
                # Format the new filename
                clean_name = f"cleaned_{selected_file}"
                
                col3, col4 = st.columns(2)
                with col3:
                    # THE DOWNLOAD BUTTON
                    st.download_button(
                        label=f"⬇️ Download {clean_name} (CSV)",
                        data=df_cleaned.to_csv(index=False).encode('utf-8'),
                        file_name=clean_name,
                        mime="text/csv",
                        use_container_width=True
                    )
                with col4:
                    # SAVE TO DATABASE BUTTON
                    if st.button(f"💾 Save {clean_name} to Database", use_container_width=True):
                        try:
                            supabase.table('cleaned_datasets').upsert({
                                'file_name': clean_name,
                                'data': df_cleaned.to_dict(orient='records')
                            }).execute()
                            st.session_state['clean_cloud'][clean_name] = df_cleaned
                            st.success("Saved to Cleaned Database!")
                        except Exception as e:
                            st.error(f"Cloud save failed: {e}")
                
                st.subheader("📊 Executive AI Report")
                eda_prompt = f"Analyze these stats: {df_cleaned.describe(include='all').to_string()}. Provide 3 trends, anomalies, and hypotheses for {domain_context}."
                st.markdown(model.generate_content(eda_prompt).text)

            except Exception as e:
                st.error(f"🚨 The AI wrote faulty code: {e}")

# --- 5. THE DATA WHISPERER (Step 4 - NEW!) ---
# Combine both raw and clean datasets so the user can chat with either!
all_chat_data = {**st.session_state.get('raw_cloud', {}), **st.session_state.get('clean_cloud', {})}

if all_chat_data:
    st.divider()
    st.header("💬 Step 4: Chat with your Data")
    
    chat_file = st.selectbox("Which dataset do you want to talk to?", list(all_chat_data.keys()))
    df_chat = all_chat_data[chat_file]
    
    # Initialize the LangChain Agent
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=st.secrets["GEMINI_API_KEY"])
    agent = create_pandas_dataframe_agent(llm, df_chat, verbose=True, allow_dangerous_code=True)
    
    # Chat UI memory
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Draw previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # The Chat Input Box
    if prompt := st.chat_input(f"Ask something about {chat_file}... (e.g., 'What is the average of column X?')"):
        # Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get AI Response
        with st.chat_message("assistant"):
            with st.spinner("Crunching the numbers..."):
                try:
                    # The Agent writes python, runs it against your dataframe, and reads the answer
                    response = agent.invoke(prompt)
                    answer = response["output"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Could not calculate that. Error: {e}")
