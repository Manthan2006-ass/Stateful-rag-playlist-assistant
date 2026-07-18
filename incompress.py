import os
import chromadb
from chromadb.utils import embedding_functions
import ollama

# Initialize the local persistent ChromaDB storage engine
DB_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings", 
    model_name="bge-m3:latest"
)

collection = chroma_client.get_collection(
    name="campusx_genai_playlist", 
    embedding_function=ollama_ef
)

def format_timestamp(seconds: float) -> str:
    """Converts raw floating-point seconds into readable MM:SS layout markers."""
    minutes = int(seconds // 60)
    remaining_secs = int(seconds % 60)
    return f"{minutes:02d}:{remaining_secs:02d}"

def condense_query(user_query: str, history: list) -> str:
    """Uses a lightweight local LLM pass to rewrite multi-turn shorthand into a standalone query."""
    if not history:
        return user_query  # First turn: search query maps straight to raw string input

    # Keep contextual evaluation focused entirely on the last two turns to preserve local CPU RAM
    history_str = ""
    for msg in history[-4:]:
        history_str += f"{msg['role'].upper()}: {msg['content']}\n"
        
    condense_prompt = (
        f"Given the following conversation history and a follow-up question, "
        f"rephrase the follow-up question into a standalone query that can be used "
        f"to search a vector database. Do NOT answer the question. Only return the rewritten query text.\n\n"
        f"Chat History:\n{history_str}\n"
        f"Follow-up Question: {user_query}\n"
        f"Standalone Query:"
    )
    
    # Use your chosen active local model (llama3.1:8b or llama3.2:3b for speed upgrades)
    response = ollama.chat(
        model="llama3.2:3b", 
        messages=[{"role": "user", "content": condense_prompt}],
        options={"temperature": 0.0}
    )
    rewritten = response["message"]["content"].strip()
    print(f"🔄 Memory Analysis Summary: '{user_query}' ──> Combined Search: '{rewritten}'")
    return rewritten

def query_rag_assistant_stream(user_query: str, history: list):
    """Performs semantic retrieval and yields streaming text chunks in real-time."""
    # 1. Condense follow-up query text against preceding session history bounds
    search_query = condense_query(user_query, history)
    
    # 2. Query the HNSW spatial indexing layer inside ChromaDB
    results = collection.query(query_texts=[search_query], n_results=4)
    
    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_meta = results.get("metadatas", [[]])[0]
    
    if not retrieved_docs:
        yield "🤖 I couldn't locate any relevant details in the playlist transcripts to resolve that."
        return

    # 3. Assemble the retrieved structural context stream
    context_stream = ""
    for idx, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_meta)):
        start_str = format_timestamp(meta["start_time"])
        end_str = format_timestamp(meta["end_time"])
        context_stream += f"\n--- Source Block {idx+1} ---\n"
        context_stream += f"Lecture {meta['video_number']}: {meta['video_title']} [{start_str} - {end_str}]\nText: {doc}\n"

    # 4. Prompt Engineering Directives
    system_prompt = (
        "You are an expert AI Teaching Assistant managing the CampusX Generative AI playlist. "
        "Your task is to answer user queries using strictly the provided course transcript context blocks. "
        "For every factual claim or code concept you mention, you MUST explicitly cite the video source "
        "and timestamp in the text format [Lecture XX at MM:SS]. Do not guess or make up info."
    )
    user_prompt = f"Context Material:\n{context_stream}\n\nStudent Question: {user_query}"

    # 5. Execute an active token streaming channel pass
    stream = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.0},
        stream=True
    )
    
    # Yield individual text components incrementally as they generate
    for chunk in stream:
        yield chunk['message']['content']