import os
import json
import chromadb
from chromadb.utils import embedding_functions
from ollama import Client as OllamaClient

# 1. Initialize the persistent local ChromaDB database directory
DB_PATH = "./chroma_db"
print(f"📦 Initializing Persistent Vector Database at: {DB_PATH}")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# 2. Setup a custom Ollama Client with an extended timeout configuration
# Pushing the network timeout window prevents 'httpx.ReadTimeout' errors on local CPUs
custom_ollama_client = OllamaClient(
    host="http://localhost:11434",
    timeout=300.0  # Gives your Intel CPU a generous 5-minute processing window
)

# 3. Configure the embedding wrapper to leverage the timeout-safe client
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="bge-m3:latest"
)
# Overwrite the internal default connection instance with our safe client
ollama_ef._client = custom_ollama_client

# Fetch or establish the vector database collection index
collection = chroma_client.get_or_create_collection(
    name="campusx_genai_playlist",
    embedding_function=ollama_ef,
    metadata={"hnsw:space": "cosine"}
)

input_folder = "newjsons"
if not os.path.exists(input_folder):
    print(f"❌ Error: '{input_folder}' directory not found. Please run merge_chunks.py first!")
    exit()

json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

print("🚀 Starting Extended-Timeout Vector Indexing Process...")

global_chunk_counter = 0
BATCH_SIZE = 5  # Sub-divides documents into small groups to protect CPU execution lanes

for json_file in json_files:
    file_path = os.path.join(input_folder, json_file)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = data.get("chunks", [])
    if not chunks:
        continue

    print(f"📥 Vectorizing & Indexing Chunks from: {json_file}")
    
    documents = []
    metadatas = []
    ids = []

    for chunk in chunks:
        documents.append(chunk["text"])
        metadatas.append({
            "video_number": chunk["number"],
            "video_title": chunk["title"],
            "start_time": float(chunk["start"]),
            "end_time": float(chunk["end"])
        })
        ids.append(f"chunk_{global_chunk_counter}")
        global_chunk_counter += 1

    # Safe Batch-Processing Loop: sends chunks in small groups
    for idx in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[idx : idx + BATCH_SIZE]
        batch_meta = metadatas[idx : idx + BATCH_SIZE]
        batch_ids = ids[idx : idx + BATCH_SIZE]
        
        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )
        except Exception as batch_error:
            print(f"⚠️ Error parsing subgroup index range [{idx}:{idx+BATCH_SIZE}]: {str(batch_error)}")

print(f"\n🎉 Success! All {global_chunk_counter} semantic blocks are successfully indexed into ChromaDB!")