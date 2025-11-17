# streamlit.py

import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from app import (
    load_dashboard_df,
    run_rag_query,
    update_llm_insights_for_filtered_df,
    UPDATED_LLM_COL,
)

st.set_page_config(page_title="Excel RAG Chatbot", layout="wide")

# ---------- CONFIG ----------
LOGO_PATH = "logo.jpg"          # your logo (optional)
CLIENT_EXCEL_FOLDER = "client_excels"  # folder with one Excel per client
# -----------------------------


# ---------- Utilities ----------

@st.cache_data
def list_client_excels(folder: str) -> dict:
    """
    List all .xlsx files in the given folder.
    Returns a dict: {display_client_name: full_file_path}
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        return {}

    mapping = {}
    for f in folder_path.iterdir():
        if f.is_file() and f.suffix.lower() in [".xlsx", ".xlsm"]:
            # Use file stem (without extension) as client display name
            client_name = f.stem
            mapping[client_name] = str(f)
    return mapping


@st.cache_data
def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "dashboard") -> bytes:
    """
    Convert a DataFrame to an in-memory Excel file for download.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.read()


# ---------- Init session state ----------

if "df_filtered" not in st.session_state:
    st.session_state.df_filtered = None
    st.session_state.df_updated_filtered = None
    st.session_state.filter_col = None
    st.session_state.filter_values = []

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []   # chat history for Q&A only

# ---------- HEADER with LOGO ----------

header_cols = st.columns([1, 5])

with header_cols[0]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.write(" ")  # empty space if logo is missing

with header_cols[1]:
    st.title("📊 RAG-based Excel Chatbot (Client-wise files)")
    st.caption("Pick a client → load that file only → chat or update LLM insights.")

st.markdown(
    """
Now each client has its **own Excel file** (one file per client).

Flow:

1. Pick a **client** from the dropdown (we read filenames from a folder).  
2. We load **only that client's Excel** into a DataFrame.  
3. You can **filter** (e.g. by Lanyon ID, City, etc.).  
4. Then choose:
   - 💬 **Chat** about the filtered data (no changes), or  
   - ✍️ **Change LLM insights** (create/update `Updated LLM insight` column).  
5. Download the **filtered + updated** rows.
"""
)

# ---------- 1) Choose client (pick which Excel file) ----------

st.subheader("1️⃣ Select client (one Excel per client)")

client_files = list_client_excels(CLIENT_EXCEL_FOLDER)

if not client_files:
    st.error(
        f"No Excel files found in folder `{CLIENT_EXCEL_FOLDER}`.\n\n"
        "Make sure you have created per-client files like `ClientA.xlsx`, "
        "`ClientB.xlsx` in that folder."
    )
    st.stop()

client_names = sorted(client_files.keys())

selected_client = st.selectbox(
    "Select client:",
    options=client_names,
    key="client_name_widget",
)

selected_client_path = client_files[selected_client]

st.info(
    f"Selected client: **{selected_client}**  \n"
    f"File: `{selected_client_path}`"
)

# ---------- 2) Load ONLY this client's Excel ----------

st.subheader("2️⃣ View data for selected client")

with st.spinner(f"Loading Excel for client `{selected_client}` ..."):
    # We override the default EXCEL_PATH by passing a specific path
    df_base = load_dashboard_df(path=selected_client_path)

if df_base.empty:
    st.warning(
        f"The Excel file for `{selected_client}` has **0 rows**. "
        "Please check the file."
    )
    st.stop()

st.write("### Data for current client")
st.dataframe(df_base, use_container_width=True)

# ---------- 3) Filter the data (within this client) ----------

st.subheader("3️⃣ Filter the data (for this client, e.g. by Lanyon ID or City)")

cols = list(df_base.columns)
if not cols:
    st.error("The DataFrame has no columns – please check your Excel file.")
    st.stop()
else:
    # Prefer Lanyon ID or City if available
    default_index = 0
    if "Lanyon ID" in cols:
        default_index = cols.index("Lanyon ID")
    elif "City" in cols:
        default_index = cols.index("City")

    filter_col_widget = st.selectbox(
        "Choose a column to filter on (for example **Lanyon ID** or **City**):",
        options=cols,
        index=default_index,
        key="filter_col_widget",
    )

    if filter_col_widget:
        possible_values = (
            df_base[filter_col_widget]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        possible_values = sorted(possible_values)
    else:
        possible_values = []

    filter_values_widget = st.multiselect(
        f"Select one or more values from **{filter_col_widget}** "
        "to keep in the filtered DataFrame:",
        options=possible_values,
        key="filter_values_widget",
    )

    # Use widget values as current filter
    filter_col = filter_col_widget
    filter_values = filter_values_widget

    # Apply filter on df_base (which is already per-client)
    if filter_values:
        df_filtered = df_base[df_base[filter_col].astype(str).isin(filter_values)].copy()
    else:
        df_filtered = df_base.copy()

    st.session_state.df_filtered = df_filtered
    st.session_state.filter_col = filter_col
    st.session_state.filter_values = filter_values

    st.write("### Filtered DataFrame (for selected client)")
    st.dataframe(df_filtered, use_container_width=True)

# ---------- 4A) OPTION 1 – 💬 Chat (Q&A only, no changes) ----------

st.subheader("4️⃣ Option 1 – 💬 Chat about the filtered data")

st.markdown(
    "Use this chat to **ask questions** about the current filtered data. "
    "This will **NOT** change the Excel file."
)

# Existing Q&A chat history
for role, content in st.session_state.qa_history:
    with st.chat_message(role):
        st.markdown(content)

qa_msg = st.chat_input(
    "Ask a question about the filtered data (no changes will be made)...",
    key="qa_input",
)

if qa_msg:
    # Show user message
    st.session_state.qa_history.append(("user", qa_msg))
    with st.chat_message("user"):
        st.markdown(qa_msg)

    df_filtered = st.session_state.df_filtered

    with st.chat_message("assistant"):
        if df_filtered is None or df_filtered.empty:
            answer = (
                "Your filtered DataFrame currently has **0 rows**. "
                "Please adjust the filters above and try again."
            )
        else:
            with st.spinner("Running RAG over the filtered data to answer your question..."):
                try:
                    answer = run_rag_query(
                        query=qa_msg,
                        df=df_filtered,      # 👈 only this client's filtered rows
                        extra_instruction=None,
                    )
                except Exception as e:
                    answer = f"Error while calling the RAG backend: `{e}`"

        st.markdown(answer)
        st.session_state.qa_history.append(("assistant", answer))

# ---------- 4B) OPTION 2 – ✍️ Change LLM insights ----------

st.subheader("5️⃣ Option 2 – ✍️ Change LLM insights for the filtered rows")

st.markdown(
    f"""
Use this section when you want to **modify the LLM insight** column for the filtered rows.

Describe how **'{UPDATED_LLM_COL}'** (or the original **'LLM insight'** column)
should be updated for these rows.
"""
)

update_instruction = st.text_area(
    f"Describe how the '{UPDATED_LLM_COL}' column should be updated for the filtered rows:",
    key="update_instruction",
    height=150,
)

if st.button("🚀 Generate new LLM insights for filtered rows"):
    df_filtered = st.session_state.df_filtered
    filter_col = st.session_state.filter_col
    filter_values = st.session_state.filter_values

    if df_filtered is None or df_filtered.empty:
        st.warning(
            "Your filtered DataFrame currently has **0 rows**. "
            "Please adjust filters above and try again."
        )
    elif not update_instruction.strip():
        st.warning("Please enter instructions for how to update the LLM insights.")
    else:
        with st.spinner(
            "Running RAG row-wise over the filtered DataFrame to update "
            f"'{UPDATED_LLM_COL}'..."
        ):
            try:
                updated_df = update_llm_insights_for_filtered_df(
                    df_filtered=df_filtered,
                    user_instruction=update_instruction,
                    filter_col=filter_col,
                    filter_values=filter_values,
                )
                st.session_state.df_updated_filtered = updated_df
                n_rows = len(updated_df)

                st.success(
                    f"Generated new **{UPDATED_LLM_COL}** values for "
                    f"**{n_rows} row(s)** in the current filtered DataFrame."
                )
            except Exception as e:
                st.error(f"Error while calling the RAG backend: `{e}`")

# ---------- 6) Review updated DF + download ----------

if st.session_state.get("df_updated_filtered") is not None:
    st.subheader("6️⃣ Review and finish")

    updated_df = st.session_state.df_updated_filtered

    st.write(
        f"#### Filtered DataFrame with new **'{UPDATED_LLM_COL}'** column (row-wise RAG output)"
    )
    st.dataframe(updated_df, use_container_width=True)

    next_action = st.radio(
        "Are you fine with these updates, or do you need more changes?",
        options=[
            "I need more changes (change filters / instructions and rerun)",
            "End this and download the updated filtered file",
        ],
        index=0,
        key="next_action_after_update",
    )

    if next_action.startswith("I need more"):
        st.info(
            "No problem. You can change the filters in step **3** or the "
            "**update instruction** in step **5**, then rerun the update. "
            f"The app will regenerate the **'{UPDATED_LLM_COL}'** column for the new filtered DataFrame."
        )
    else:
        st.success(
            "Great! Click the button below to download the updated file containing "
            "only the **filtered rows** for this client plus the new "
            f"**'{UPDATED_LLM_COL}'** column."
        )

        excel_bytes_updated = df_to_excel_bytes(
            updated_df, sheet_name=f"{selected_client}_filtered_with_updated_llm"
        )

        st.download_button(
            label="⬇️ Download updated filtered Excel",
            data=excel_bytes_updated,
            file_name=f"{selected_client}_filtered_updated_llm_insight.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )
