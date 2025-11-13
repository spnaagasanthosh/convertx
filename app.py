 '''Streamlit front-end to convert Oracle SQL/PLSQL to Databricks-ready code
  using a Databricks Serving Endpoint (LLM). Requires `streamlit` and
  `databricks-sdk`.'''
  

import base64
import os
from dataclasses import dataclass
from typing import List, Optional
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------

@dataclass
class ConversionResult:
    oracle_snippet: str
    databricks_output: str
    model_name: str
    tokens_used: Optional[int] = None
    notes: Optional[str] = None


# ------------------------------------------------------------------------------
# Databricks helpers
# ------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_workspace_client(host: str, token: str) -> WorkspaceClient:
    """Create a Databricks WorkspaceClient. Cached so repeated calls are fast."""
    return WorkspaceClient(host=host, token=token)


def call_llm(workspace: WorkspaceClient, endpoint: str, oracle_sql: str, system_prompt: str) -> dict:
    """Call the Databricks LLM endpoint and return the raw completion payload."""
    return workspace.serving_endpoints.completions.create(
        name=endpoint,
        input={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": oracle_sql},
            ]
        },
    )


# ------------------------------------------------------------------------------
# Streamlit layout
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Oracle ➜ Databricks Translator", layout="wide")
st.title("Oracle SQL/PLSQL ➜ Databricks Translator")

# Session state to keep history
if "history" not in st.session_state:
    st.session_state.history: List[ConversionResult] = []

# Sidebar config ---------------------------------------------------------------
with st.sidebar:
    st.header("Databricks Connection")
    default_host = os.environ.get("DATABRICKS_HOST", "")
    default_token = os.environ.get("DATABRICKS_TOKEN", "")
    host = st.text_input("Workspace URL", value=default_host, placeholder="https://adb-xxx.azuredatabricks.net")
    token = st.text_input("Personal Access Token", value=default_token, type="password")
    endpoint_name = st.text_input("Serving Endpoint Name", value="databricks-dbrx-instruct")

    st.markdown("---")
    st.subheader("Target Metadata (optional)")
    target_catalog = st.text_input("Catalog", value="main")
    target_schema = st.text_input("Schema", value="default")

    if st.button("Test Connection"):
        if not host or not token:
            st.warning("Host and token are required.")
        else:
            try:
                client = get_workspace_client(host, token)
                me = client.current_user.me()
                st.success(f"Connected as {me.user_name}")
            except DatabricksError as exc:
                st.error(f"Connection failed: {exc}")


# Input section ----------------------------------------------------------------
st.subheader("Oracle Source Objects")
tab_manual, tab_file, tab_examples = st.tabs(["Paste Text", "Upload File", "Examples"])

with tab_manual:
    oracle_text = st.text_area(
        "Oracle SQL/PLSQL",
        height=280,
        placeholder="CREATE OR REPLACE PROCEDURE ...",
    )

with tab_file:
    upload = st.file_uploader("Upload .sql/.ddl/.pls file", type=["sql", "ddl", "pls", "txt"])
    if upload:
        oracle_text = upload.getvalue().decode("utf-8")

with tab_examples:
    sample = st.selectbox(
        "Example objects",
        [
            "Choose…",
            "Table + sequence",
            "Package spec/body",
            "Cursor-based procedure",
        ],
    )
    samples = {
        "Table + sequence": """CREATE TABLE employees (...);""",
        "Package spec/body": """CREATE OR REPLACE PACKAGE ...;""",
        "Cursor-based procedure": """CREATE OR REPLACE PROCEDURE ...;""",
    }
    if sample in samples:
        oracle_text = samples[sample]
        st.code(oracle_text, language="sql")

notes = st.text_input("Migration notes / context (optional)")
object_name = st.text_input("Target object name", placeholder="e.g. stg_orders_proc")

system_prompt = f"""
You are a Databricks migration assistant.
Convert Oracle SQL/PLSQL into Spark SQL or PySpark code targeting catalog `{target_catalog}` schema `{target_schema}`.
Explain assumptions and handling of transactions/cursors/exceptions if present.
"""

convert_disabled = not oracle_text or not host or not token or not endpoint_name

# Conversion action ------------------------------------------------------------
if st.button("Convert", disabled=convert_disabled):
    with st.spinner("Calling Databricks LLM..."):
        try:
            client = get_workspace_client(host, token)
            completion = call_llm(client, endpoint_name, oracle_text, system_prompt)

            candidate = completion.predictions[0].get("candidates", [{}])[0]
            message = candidate.get("message", {})
            response_text = message.get("content", "")
            token_usage = completion.predictions[0].get("usage", {}).get("total_tokens")

            if not response_text:
                raise ValueError("LLM returned empty content.")

            result = ConversionResult(
                oracle_snippet=oracle_text,
                databricks_output=response_text,
                model_name=endpoint_name,
                tokens_used=token_usage,
                notes=notes,
            )
            st.session_state.history.append(result)
            st.success("Conversion complete.")
        except Exception as exc:
            st.error(f"Conversion failed: {exc}")

# Output area ------------------------------------------------------------------
if st.session_state.history:
    st.subheader("Latest Conversion")
    latest = st.session_state.history[-1]
    st.caption(f"Endpoint `{latest.model_name}` • Tokens: {latest.tokens_used or 'N/A'}")
    st.code(latest.databricks_output, language="sql")

    cols = st.columns(3)
    with cols[0]:
        st.download_button(
            "Download SQL",
            latest.databricks_output.encode("utf-8"),
            file_name=f"{object_name or 'conversion'}.sql",
        )
    with cols[1]:
        if st.button("Copy to Clipboard", key="copy_btn"):
            st.toast("Use the clipboard icon on the code block to copy.")
    with cols[2]:
        if st.button("Show Oracle Source", key="show_source"):
            st.code(latest.oracle_snippet, language="sql")

    st.markdown("### Conversion History")
    for idx, item in reversed(list(enumerate(st.session_state.history))):
        with st.expander(f"Run #{idx+1} – {item.model_name}"):
            st.write(f"Tokens: {item.tokens_used or 'N/A'}")
            if item.notes:
                st.write(f"Notes: {item.notes}")
            st.code(item.databricks_output, language="sql")

else:
    st.info("Run a conversion to see results here.")
