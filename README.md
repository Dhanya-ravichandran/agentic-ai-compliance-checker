### Agentic AI Compliance Checker (Agentic RAG)

**Overview**
This project implements an **Agentic Retrieval-Augmented Generation (RAG)** system that evaluates whether internal company policies comply with regulatory frameworks such as **GDPR and the EU AI Act**. The system autonomously retrieves, compares, verifies, and explains compliance gaps using multiple AI agents.

**Key Features**

* Dynamic PDF ingestion (Law + Company Policy)
* Agent-based decision-making and tool selection
* Source-cited compliance analysis
* Self-verification using a Critic agent
* Interactive Streamlit interface

**Agent Workflow**

1. **Planner Agent** – Determines which tools are required for the query
2. **Law Retriever Agent** – Retrieves relevant regulation clauses
3. **Policy Retriever Agent** – Retrieves company policy sections
4. **Web Search Agent** – Fetches latest regulatory updates when needed
5. **Compliance Agent** – Compares laws vs policies and identifies gaps
6. **Critic Agent** – Reviews reasoning quality and triggers retries if required

**Tech Stack**

* Python
* LangGraph (Agent orchestration)
* Retrieval-Augmented Generation (RAG)
* Pinecone (Vector Database)
* Sentence-Transformers (Embeddings)
* Groq LLM (LLaMA 3.1)
* Tavily (Web Search)
* Streamlit (UI)

**Use Case**
Applicable for **compliance audits, regulatory analysis, Responsible AI assessments**, and **policy governance automation** in regulated industries.

