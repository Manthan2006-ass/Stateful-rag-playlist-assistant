# 🎓 CampusX GenAI Playlist Assistant (Stateful Hybrid RAG Architecture)

An end-to-end, high-performance Retrieval-Augmented Generation (RAG) platform designed to act as an intelligent Teaching Assistant over technical audio-visual course materials. 

## ⚡ Core Engineering Features
- **Stateful Query Condensation:** Leverages local contextual history evaluation to rewrite follow-up conversational shorthand, completely eliminating multi-turn pronoun ambiguities.
- **Logarithmic Spatial Retrieval ($O(\log N)$):** Replaced slow, linear context scanning structures with a persistent **ChromaDB** container running **HNSW graphing algorithms** to maintain sub-millisecond retrieval bounds.
- **Cross-Lingual Information Routing:** Built using the multi-granularity **BGE-M3 Multilingual Embedding Foundation Model** to accurately map mixed technical terminology across Hindi conversational tones and English code syntax.
- **Token Streaming Engine:** Bypasses CPU inference delays by implementing a dynamic chunk generator payload channel, streaming tokens immediately to the UI layer.

## 🛠️ Technology Stack
- **Frontend Layer:** Streamlit (Enterprise Dashboard UI with Citation Badging)
- **Gateway APIs:** FastAPI (Streaming Web Gateway)
- **Vector Base Engine:** ChromaDB (Hierarchical Navigable Small World Indexing)
- **Embedding Matrix:** BGE-M3 (via Local Ollama API endpoints)
- **Neural Compute Engine:** Llama 3.2 3B Parameters (Edge-Computed via Ollama)

## 🚀 How to Run Locally

1. **Pull the Local Foundation Models:**
   ```bash
   ollama pull bge-m3:latest
   ollama pull llama3.2:3b