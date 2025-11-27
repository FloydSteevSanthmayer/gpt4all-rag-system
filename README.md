# GPT4All RetrievalQA (Streamlit + Chroma)

**Local Retrieval-Augmented QA application** built with Streamlit, LangChain, a local GPT4All GGUF model and Chroma vectorstore.

## What's included
- `app.py` — Streamlit app (upload PDF, build vectorstore, query LLM)
- `download_mistral_model.py` — Robust downloader to fetch GGUF model from Hugging Face
- `requirements.txt` — pinned dependencies (recommended)
- `.gitignore` — ignores large model files & local caches
- `FLOWCHART.md` — diagram + explanation of app flow
- `LICENSE` — MIT license (copyright: Floyd Steev Santhmayer)

## Quick start (local / on-prem)
1. Create & activate a virtual environment:
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the GGUF model (example):
```bash
python download_mistral_model.py --url "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_0.gguf" --out "C:\\models\\mistral-7b-openorca.Q4_0.gguf" --token "hf_xxx"
```

4. Run the app:
```bash
streamlit run app.py
```

## Notes
- Model files are very large; keep them off the repository. Add them to `.gitignore` (already included).
- Persisted Chroma databases are stored per-uploaded-PDF under `chroma_storage/`.
- If you use PyCharm, point the project interpreter to your `.venv` and Invalidate Caches / Restart if imports show as unresolved.

---
