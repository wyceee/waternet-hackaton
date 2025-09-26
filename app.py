"""Streamlit UI for exploring the water quality CSV with LLM assistance."""

import os
import pandas as pd
import streamlit as st
from water_core import (
    init_openai,
    load_water_data,
    filter_water_data,
    llm_suggest_filters,
    build_prompt,
    ask_llm,
)

st.set_page_config(page_title="Water Quality QA", layout="wide")
st.title("💧 Water Quality Data Explorer + LLM")

@st.cache_data(show_spinner=False)
def cached_load_csv(path: str) -> pd.DataFrame:
    return load_water_data(path)


@st.cache_data(show_spinner=False)
def cached_upload_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=';')
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    return df


def get_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    if os.path.exists("openai.key"):
        try:
            with open("openai.key") as f:
                return f.read().strip()
        except Exception:
            pass
    return None


with st.sidebar:
    st.header("Configuration")
    existing_key = get_api_key()
    user_key = st.text_input("OpenAI API Key", value=existing_key or "", type="password")
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-5"))
    if user_key and user_key != existing_key:
        os.environ["OPENAI_API_KEY"] = user_key
    if model:
        os.environ["OPENAI_MODEL"] = model
    if user_key:
        try:
            init_openai(user_key)
            st.success("OpenAI client ready.")
        except Exception as e:
            st.error(f"Failed to init client: {e}")
    else:
        st.warning("Enter an API key to enable model calls.")

    st.markdown("---")
    st.caption("Upload a custom CSV (semicolon separated) or use the bundled sample.")
    uploaded = st.file_uploader("CSV Upload", type=["csv"])
    # Automatically discover CSV files in the repository root
    try:
        csv_files = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
    except Exception:
        csv_files = []
    st.caption("Detected repository CSV files (select one or more):")
    selected_files: list[str] = []
    # Default: if no upload, preselect the first discovered file
    for idx, fname in enumerate(csv_files):
        default_checked = uploaded is None and idx == 0
        if st.checkbox(fname, value=default_checked, key=f"csv_select_{fname}"):
            selected_files.append(fname)
    if not csv_files:
        st.info("No CSV files found in repository root.")
    st.caption("If multiple files are selected they will be concatenated (row-wise).")
    st.markdown("---")
    st.subheader("Manual Filters")
    species = st.text_input("Species contains")
    location = st.text_input("Location contains")
    parameter = st.text_input("Parameter contains")
    year = st.number_input("Year", min_value=0, max_value=3000, step=1, value=0)
    year_val = int(year) if year else None
    st.markdown("---")
    max_rows = st.slider("Max rows in prompt", min_value=5, max_value=1000, value=250, step=5)

df: pd.DataFrame | None = None
# Priority: uploaded file (if provided), else selected repository CSV files.
if 'uploaded' in locals() and uploaded is not None:
    try:
        df = cached_upload_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
elif selected_files:
    loaded_frames = []
    for fpath in selected_files:
        try:
            loaded_frames.append(cached_load_csv(fpath))
        except Exception as e:
            st.error(f"Failed to load {fpath}: {e}")
    if loaded_frames:
        if len(loaded_frames) == 1:
            df = loaded_frames[0]
        else:
            try:
                df = pd.concat(loaded_frames, ignore_index=True)
            except Exception as e:
                st.error(f"Failed to concatenate selected CSVs: {e}")
else:
    st.info("Upload a CSV or select at least one detected repository CSV.")

if df is not None:
    st.subheader("Dataset Preview")
    st.write(f"Total rows: {len(df):,}")
    # Streamlit deprecation: use_container_width -> width ('stretch'|'content').
    # Keep backward compatibility for older versions by falling back if needed.
    def _dataframe_full_width(data):
        try:
            return st.dataframe(data, width="stretch")
        except TypeError:
            return st.dataframe(data, use_container_width=True)  # older Streamlit

    _dataframe_full_width(df.head(500))
    st.markdown("---")
    st.subheader("Ask a Question")
    question = st.text_area("Your question about the data", placeholder="e.g. What species were found at volkstuinen Amstelglorie?", height=100)
    col_a, col_b = st.columns([1,1])
    with col_a:
        use_manual = st.checkbox("Apply manual filters before asking", value=False)
    with col_b:
        run = st.button("Ask Model", type="primary")
    if run and question.strip():
        if not get_api_key():
            st.error("API key required.")
        else:
            try:
                if use_manual:
                    manual_df = filter_water_data(df, species or None, location or None, year_val or None, parameter or None)
                else:
                    manual_df = df
                st.write(f"Manual filter rows: {len(manual_df)}")
                with st.spinner("Inferring filters with LLM..."):
                    inferred = llm_suggest_filters(question, manual_df.columns.tolist())
                st.write("Inferred filters:", inferred)
                auto_df = filter_water_data(manual_df, inferred.get("species"), inferred.get("location"), inferred.get("year"), inferred.get("parameter"))
                st.write(f"Rows after inferred filters: {len(auto_df)}")
                with st.expander("Filtered Data Preview"):
                    _dataframe_full_width(auto_df.head(100))
                prompt = build_prompt(question, auto_df, max_rows=max_rows)
                with st.expander("Prompt (debug)"):
                    st.code(prompt)
                with st.spinner("Querying model..."):
                    answer = ask_llm(prompt)
                st.success("Answer")
                st.markdown(answer)
            except Exception as e:
                st.error(f"Error: {e}")
    elif run:
        st.warning("Enter a question first.")
