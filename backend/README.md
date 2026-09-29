# Document Question with RAG — FastAPI Backend

This backend converts the supplied Streamlit RAG logic into HTTP API endpoints while preserving its core concepts:
- OpenAI `gpt-4o-mini` answer model
- OpenAI `text-embedding-3-small` embeddings
- Persistent Chroma database at `./chroma_db`
- PDF text extraction with `pdfplumber`
- OCR fallback for scanned PDFs using PyMuPDF, Tesseract, and Pillow
- Chunk size 1000 and overlap 200
- Generated overview based on the first five extracted pages, stored as `type="overview"`
- Overview-query keyword detection and stored-overview lookup
- Similarity search with configurable top-k and maximum-distance threshold
- Original RAG system prompt and fallback answer
- Database clear endpoint
- Debug response containing scores and retrieved documents

## Setup (Windows PowerShell)

1. Create/activate a virtual environment.
2. Install packages:

   `python -m pip install -r requirements.txt`

3. Copy `.env.example` to `.env` and set your OpenAI API key.
4. Start the API:

   `uvicorn main:app --reload --port 8000`

5. Open:
   - Health: http://127.0.0.1:8000/health
   - API docs: http://127.0.0.1:8000/docs

## Endpoints

- `GET /`
- `GET /health`
- `GET /api/settings`
- `POST /api/documents` — multipart form field `file`
- `POST /api/chat` — JSON `question`, optional `top_k`, `relevance_threshold`, `temperature`
- `DELETE /api/database` — clear the Chroma collection

## Chat request example

```json
{
  "question": "What is this document about?",
  "top_k": 4,
  "relevance_threshold": 2.0,
  "temperature": 0.0
}
```

## Notes

- Tesseract OCR is a separate system application and must be installed for scanned PDFs.
- The Windows fallback path is retained from the original script. On Linux hosting, install Tesseract and ensure `tesseract` is on PATH.
- Add your deployed Next.js frontend origin to `allow_origins` before production use.
- The local Chroma directory is not durable on many ephemeral hosting services unless persistent storage is configured.
- This API intentionally uses a single Chroma collection, matching the original Streamlit app's behaviour of replacing the old collection when processing a new PDF.
