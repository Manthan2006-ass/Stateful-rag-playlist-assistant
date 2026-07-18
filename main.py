import streamlit as st
import requests
import re

# 1. Premium Page Configuration Layout
st.set_page_config(
    page_title="CampusX GenAI Assistant",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# 2. Professional Enterprise Sidebar Design
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/artificial-intelligence.png", width=80)
    st.title("System Controls")
    st.caption("CampusX Generative AI Playlist Copilot")
    st.markdown("---")
    
    # Live Status Indicators for Interview Portfolio Value
    st.markdown("### ⚡ System Status")
    st.success("Vector Engine: ChromaDB (Active)")
    st.success("Model Layer: Llama 3.2 3B (Local)")
    st.info("Embedding: BGE-M3 Multilingual")
    
    st.markdown("---")
    
    # Utility Operational Features
    if st.button("🔄 Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("⚙️ *Designed for High-Scalability Local Video Retrieval RAG Pipelines.*")

# 3. Main Dashboard Header Layout
st.title("🎓 CampusX AI Teaching Assistant")
st.markdown("Ask deep technical questions regarding the Generative AI Course Playlist lectures. The assistant handles multi-turn context and prints precise source text citations automatically.")
st.markdown("---")

# Initialize chat session log buffers globally if empty
if "messages" not in st.session_state:
    st.session_state.messages = []

def display_citations_only(text: str):
    """
    Exclusively scans response content for structural citation targets 
    and formats them cleanly into UI grid cards without double-printing the text body.
    """
    # Extract unique structural citation targets: [Lecture XX at MM:SS]
    matches = re.findall(r'\[Lecture\s+(\d+)\s+at\s+(\d+):(\d+)\]', text)
    
    if matches:
        st.markdown("#### 📚 Reference Citations")
        unique_matches = sorted(list(set(matches)))
        
        # Create a neat grid layout using dynamic column allocations
        cols = st.columns(max(len(unique_matches), 1))
        for index, match in enumerate(unique_matches):
            lec_num = match[0]
            minutes = match[1]
            seconds = match[2]
            
            with cols[index]:
                # Wrap each reference location inside a stylized card layout container
                with st.container(border=True):
                    st.markdown(f"**🎬 Lecture {lec_num}**")
                    st.caption(f"⏱️ Marker: `{minutes}:{seconds}`")

# 4. Main Chat History Rendering Loop
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])  # Renders core message text cleanly
        if msg["role"] == "assistant":
            display_citations_only(msg["content"])  # Appends citation dashboard cards strictly below it

# 5. User Conversation Input Layer
user_input = st.chat_input("Ask a question about LangChain, embeddings, or agent architectures...")

if user_input:
    # Commit user prompt to active layout view and message buffers
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Package context parameters to match the backend FastAPI schema requirements
    payload = {
        "question": user_input,
        "history": st.session_state.messages[:-1]  # Binds existing chat log array to date
    }

    # Request the model stream response block from the backend gateway
    with st.chat_message("assistant"):
        try:
            res = requests.post(
                "http://127.0.0.1:8000/query",
                json=payload,
                stream=True,  # Keeps rapid chunk streaming network lane active
                timeout=300
            )
            
            # 1. Animate tokens text-by-text as they arrive over the stream pipeline
            answer = st.write_stream(res.iter_content(decode_unicode=True))
            
            # 2. Draw beautiful reference metric cards cleanly below the final output string
            st.markdown("---")
            display_citations_only(answer)
            
        except Exception as e:
            answer = f"❌ Backend Pipeline Connection Error: {e}"
            st.error(answer)

    # Commit the final aggregated answer string to state storage tracking buffers
    st.session_state.messages.append({"role": "assistant", "content": answer})