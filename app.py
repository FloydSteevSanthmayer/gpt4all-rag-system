import os
import hashlib
import tempfile
import logging
from typing import List

import streamlit as st

# LangChain core
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.manager import CallbackManager

# LangChain-community (local LLM, embeddings, loaders, chroma wrapper)
from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# chromadb client (explicit client creation helps IDE type-checkers)
import chromadb
from chromadb.config import Settings

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Configuration (edit as needed)
# -------------------------
# Path to a local GGUF model file (update to your local path)
MODEL_PATH = r"C:/Users/LAP14/Downloads/mistral-7b-openorca.Q4_0.gguf"

# Base folder for storing chroma DBs; we'll create one subfolder per-PDF using a hash
BASE_PERSIST_DIR = "chroma_storage"

# Chroma backend
CHROMA_IMPL = "duckdb+parquet"  # good default for local use

# Text splitting config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0

# Retriever config
RETRIEVER_K = 4

# -------------------------
# Utility helpers
# -------------------------
def filename_hash(name: str) -> str:
    """Return a short hash for a filename to use as a persistent folder name."""
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return h[:12]


def make_persist_dir_for_file(filename: str) -> str:
    """Return an on-disk folder path for a given filename (unique per-file)."""
    h = filename_hash(filename)
    return os.path.join(BASE_PERSIST_DIR, h)


# -------------------------
# Cached resources
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load and cache the GPT4All LLM wrapper."""
    callback_manager = CallbackManager([])
    try:
        llm = GPT4All(model=model_path, callback_manager=callback_manager, verbose=False)
        logger.info("Loaded GPT4All model from %s", model_path)
        return llm
    except Exception as e:
        logger.exception("Failed to load GPT4All model: %s", e)
        raise


@st.cache_resource(show_spinner=False)
def build_vectorstore_for_pdf(pdf_path: str, persist_directory: str):
    """
    Build (and persist) a Chroma vectorstore for the provided PDF file.
    Returns a Chroma vectorstore instance.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into text chunks
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = splitter.split_documents(documents)

    # Embeddings (local GPT4AllEmbeddings wrapper)
    embeddings = GPT4AllEmbeddings()

    # Create an explicit chromadb client so IDE/type-checkers are happy
    os.makedirs(persist_directory, exist_ok=True)
    client_settings = Settings(chroma_db_impl=CHROMA_IMPL, persist_directory=persist_directory)
    client = chromadb.Client(client_settings)

    # Some versions of the wrapper expect different param names (embedding vs embeddings); try both
    try:
        vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, client=client, persist_directory=persist_directory)
    except TypeError:
        # fallback to older signature
        vectorstore = Chroma.from_documents(texts, embeddings, client=client, persist_directory=persist_directory)

    # Explicit persist if supported
    try:
        vectorstore.persist()
    except Exception:
        # ignore if not supported by the particular wrapper version
        pass

    logger.info("Built vectorstore at %s (from %s)", persist_directory, pdf_path)
    return vectorstore


# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title="Local RetrievalQA (GPT4All + Chroma)", layout="wide")
    st.title("Local RetrievalQA — Upload PDF & Ask Questions")

    # Sidebar controls
    st.sidebar.header("Settings")
    st.sidebar.info("Upload a PDF, then click 'Build/Use Vectorstore'.")
    model_path = st.sidebar.text_input("Local model path (GGUF)", value=MODEL_PATH)
    rebuild_vectorstore = st.sidebar.button("(Re)Build vectorstore from uploaded PDF")

    # Upload area (PDF required)
    st.subheader("PDF Upload (required)")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is None:
        st.warning("Please upload a PDF to proceed.")
        st.stop()

    # Save uploaded file into a temp path so PyPDFLoader and other tools can read it
    temp_dir = tempfile.gettempdir()
    saved_name = f"uploaded_{uploaded_file.name}"
    saved_path = os.path.join(temp_dir, saved_name)
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded and saved to temporary file: {saved_path}")
    pdf_path_to_use = saved_path

    # Build a persist directory specific to this PDF (so multiple PDFs can coexist)
    persist_dir = make_persist_dir_for_file(os.path.basename(pdf_path_to_use))
    persist_directory = os.path.join(BASE_PERSIST_DIR, persist_dir)

    # Show path info and actions
    with st.expander("Vectorstore & model info", expanded=False):
        st.write("Model path:", model_path)
        st.write("PDF path:", pdf_path_to_use)
        st.write("Chroma persist directory:", persist_directory)

    # Build or load vectorstore
    vectorstore = None
    if rebuild_vectorstore:
        # Clear cached vectorstore for this PDF (so it rebuilds)
        try:
            if os.path.exists(persist_directory):
                st.warning("Removing existing persisted vectorstore folder so a fresh one will be created.")
                # remove files inside persist_directory
                for root, dirs, files in os.walk(persist_directory, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception:
                            pass
                try:
                    os.rmdir(persist_directory)
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Failed to clear existing persist directory: {e}")

        # Build the vectorstore (cached function will store it)
        with st.spinner("Building vectorstore (this can take a while)..."):
            try:
                vectorstore = build_vectorstore_for_pdf(pdf_path_to_use, persist_directory)
                st.success("Vectorstore built and persisted.")
            except Exception as e:
                st.exception(f"Failed to build vectorstore: {e}")
                st.stop()
    else:
        # Try to load existing vectorstore if present; if not, build it
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            # Attempt to instantiate the Chroma wrapper that uses the existing persisted DB
            try:
                client_settings = Settings(chroma_db_impl=CHROMA_IMPL, persist_directory=persist_directory)
                client = chromadb.Client(client_settings)
                # Try both signatures
                try:
                    vectorstore = Chroma(persist_directory=persist_directory, client=client, embedding_function=None)
                except TypeError:
                    # fallback: load from_documents with empty list and existing client (some wrappers behave differently)
                    vectorstore = Chroma.from_documents([], embedding=GPT4AllEmbeddings(), client=client, persist_directory=persist_directory)
                st.info("Loaded existing persisted vectorstore.")
            except Exception as e:
                st.warning("Could not auto-load persisted vectorstore; rebuilding instead.")
                with st.spinner("Building vectorstore..."):
                    try:
                        vectorstore = build_vectorstore_for_pdf(pdf_path_to_use, persist_directory)
                        st.success("Vectorstore built and persisted.")
                    except Exception as e2:
                        st.exception(f"Failed to build vectorstore: {e2}")
                        st.stop()
        else:
            # No existing DB — build it
            with st.spinner("Building vectorstore for the first time..."):
                try:
                    vectorstore = build_vectorstore_for_pdf(pdf_path_to_use, persist_directory)
                    st.success("Vectorstore built and persisted.")
                except Exception as e:
                    st.exception(f"Failed to build vectorstore: {e}")
                    st.stop()

    # Load model
    try:
        llm = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load local model: {e}")
        st.stop()

    # Build retriever & QA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",  # safer for long docs
        retriever=retriever,
        return_source_documents=True,
    )

    # Query UI
    st.subheader("Ask a question")
    query = st.text_input("Enter your question about the uploaded PDF:")
    if st.button("Ask"):
        if not query or not query.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Retrieving & answering..."):
                answer_text = None
                src_docs: List = []
                # Try call styles that different langchain versions support
                try:
                    # Preferred: call the chain to get a dict result (includes sources)
                    result = qa({"query": query})
                    if isinstance(result, dict):
                        # Many RetrievalQA implementations put the answer in "result" or "answer"
                        answer_text = result.get("result") or result.get("answer") or str(result)
                        src_docs = result.get("source_documents") or result.get("source_documents", []) or []
                    else:
                        # result might be a string
                        answer_text = str(result)
                except Exception as e1:
                    # Fallback to .run() which returns text in many versions
                    try:
                        answer_text = qa.run(query)
                    except Exception as e2:
                        st.error(f"Chain execution failed: {e1} / {e2}")
                        st.stop()

                st.subheader("Answer")
                st.write(answer_text)

                # If we haven't obtained source docs yet, try calling qa directly a second way
                if not src_docs:
                    try:
                        raw = qa({"query": query})
                        if isinstance(raw, dict):
                            src_docs = raw.get("source_documents", [])
                    except Exception:
                        src_docs = []

                # Display sources if available
                if src_docs:
                    st.subheader("Source snippets")
                    for i, doc in enumerate(src_docs, start=1):
                        meta = getattr(doc, "metadata", {}) or {}
                        source = meta.get("source", "unknown")
                        page = meta.get("page", "n/a")
                        st.markdown(f"**Source {i}** — {source} (page: {page})")
                        snippet = getattr(doc, "page_content", str(doc))[:900]
                        st.write(snippet + ("..." if len(snippet) >= 900 else ""))
                else:
                    st.info("No source snippets available to display for this run.")

    # Footer action: allow user to clear persisted vectorstore for this PDF
    st.markdown("---")
    if st.button("Delete persisted vectorstore for this PDF"):
        if os.path.exists(persist_directory):
            try:
                # delete files (best-effort)
                for root, dirs, files in os.walk(persist_directory, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception:
                            pass
                try:
                    os.rmdir(persist_directory)
                except Exception:
                    pass
                st.success("Persisted vectorstore removed. Rebuild it using the button above.")
            except Exception as e:
                st.error(f"Failed to delete persisted vectorstore: {e}")
        else:
            st.info("No persisted vectorstore found for this PDF.")

    st.write("Tip: If Streamlit or PyCharm complains about imports, ensure the project venv has required packages and re-index the IDE.")


if __name__ == "__main__":
    main()
