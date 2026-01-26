# Normal Question:
    # "Does the company comply with AI risk management requirements under the EU AI Act?"

# Alternative Web-search-triggering questions
    # Have there been any recent updates, enforcement guidelines, or penalties related to AI risk management under the EU AI Act that could affect the company’s compliance?
    # “Are there any recent EU AI Act enforcement actions or regulatory guidance issued after 2024 that impact AI risk management obligations?”
    # "Have any companies been fined or warned under the EU AI Act for inadequate AI risk management practices?”
    # “Are there recent interpretations or clarifications by EU regulators regarding AI risk management requirements?”

# To Trigger the critic loop:
    # Cite the exact policy sections and EU AI Act Article 9 clauses that demonstrate whether a documented AI risk management system exists. If evidence is missing, explicitly state ‘NO EVIDENCE FOUND’.

import streamlit as st
from document_loader import load_uploaded_pdfs
from chunk import chunk_documents
from pinecone_upsert import upsert_to_pinecone
from compliance_graph import build_graph

mermaid_diagram = """
flowchart TD
    Q[User Query] --> P[🧠 Planner Agent]

    P --> L[📚 Law Retriever]
    P --> C[📄 Policy Retriever]
    P --> W[🌐 Web Search Agent]

    L --> A[⚖️ Compliance Analysis Agent]
    C --> A
    W --> A

    A --> R{🧪 Critic Agent}

    R -- PASS --> END[✅ Final Answer]
    R -- RETRY --> P
"""

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Compliance Checker",
    layout="wide"
)

# =============================
# Header
# =============================

st.markdown(
    """
    <div style="text-align: center;">
        <h1>Compliance Checker with Agentic RAG</h1>
        <p style="font-size: 1.05rem; color: #6c757d;">
            An Agentic RAG system for analyzing compliance between regulations
            and internal company policies using AI agents.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# =============================
# Step 1: Upload Documents
# =============================
st.subheader(" Step 1 — Upload Documents")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        law_pdf = st.file_uploader(
            "Law / Regulation (GDPR, EU AI Act)",
            type=["pdf"],
            key="law_pdf"
        )

    with col2:
        policy_pdf = st.file_uploader(
            "Company Policy Document",
            type=["pdf"],
            key="policy_pdf"
        )

    process_clicked = st.button("📥 Process & Index Documents", use_container_width=True)

    if process_clicked:
        if not law_pdf or not policy_pdf:
            st.warning("Please upload both a law document and a company policy.")
        else:
            with st.spinner("Processing, chunking, embedding, and indexing documents..."):
                law_docs = load_uploaded_pdfs(law_pdf, source_type="law")
                policy_docs = load_uploaded_pdfs(policy_pdf, source_type="policy")

                law_chunks = chunk_documents(law_docs)
                policy_chunks = chunk_documents(policy_docs)

                upsert_to_pinecone(law_chunks, namespace="law")
                upsert_to_pinecone(policy_chunks, namespace="policy")

            st.success("Documents indexed successfully. You may now ask compliance questions.")

st.divider()

# =============================
# Load LangGraph (Once)
# =============================
@st.cache_resource
def load_graph():
    return build_graph()

app = load_graph()

# =============================
# Step 2: Ask Question
# =============================
st.subheader(" Step 2 — Ask a Compliance Question")

query = st.text_area(
    "Compliance Question",
    placeholder="Does the company comply with AI risk management requirements under the EU AI Act?",
    height=100
)

run_clicked = st.button(" Run Compliance Analysis", use_container_width=True)

# =============================
# Step 3: Results
# =============================
if run_clicked and query.strip():
    with st.spinner("Running agentic compliance analysis..."):
        result = app.invoke({"query": query})

    st.divider()

    # -------------------------
    # Compliance Result
    # -------------------------
    st.subheader("📊 Compliance Assessment")
    st.markdown(result.get("analysis", "No analysis generated."))

    # -------------------------
    # Agent Execution Flow
    # -------------------------
    st.subheader("🧭 Agentic RAG Execution Flow")
    st.components.v1.html(
        f"""
        <div class="mermaid">
        {mermaid_diagram}
        </div>

    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
    </script>
    """,
    height=450
    )

    st.subheader("Agent Execution Flow")

    trace = result.get("agent_trace", [])
    if trace:
        for step in trace:
            st.markdown(f"- {step}")
    else:
        st.info("No agent execution trace available.")

    # -------------------------
    # Evidence & Sources
    # -------------------------
    st.subheader("Evidence Used")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Law References**")
        law_sources = result.get("law_sources", [])
        if law_sources:
            for src in law_sources:
                st.markdown(f"- {src}")
        else:
            st.caption("No law references recorded.")

    with col2:
        st.markdown("**Policy References**")
        policy_sources = result.get("policy_sources", [])
        if policy_sources:
            for src in policy_sources:
                st.markdown(f"- {src}")
        else:
            st.caption("No policy references recorded.")

    # -------------------------
    # Critic Review
    # -------------------------
    st.subheader("Critic Evaluation")

    decision = result.get("critic_decision", "N/A")
    feedback = result.get("critic_feedback", "No critic feedback available.")

    if decision == "PASS":
        st.success(f"Decision: {decision}")
    elif decision == "RETRY":
        st.error(f"Decision: {decision}")
    else:
        st.info(f"Decision: {decision}")

    st.markdown(feedback)
#venv\Scripts\Activate
#Streamlit run Streamlit_app.py