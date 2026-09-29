import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import fitz
import pdfplumber
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Preserve the original Windows Tesseract fallback.
if not shutil.which("tesseract"):
    default_tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(default_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = default_tesseract_path


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(title="Document Question with RAG")

# Local Next.js development frontend. Add your deployed frontend
# origin to this list when deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://pdf-rag-system-yomp.vercel.app",

    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# LOAD CHROMA VECTOR DATABASE
# =========================================================

CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def load_vectorstore() -> Chroma:
    """Open the same persistent Chroma database used by the original app."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY or None,
    )
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


try:
    vectorstore = load_vectorstore()
except Exception as exc:
    # Keep the API importable so /health and /docs remain available.
    vectorstore = None
    VECTORSTORE_STARTUP_ERROR = str(exc)
else:
    VECTORSTORE_STARTUP_ERROR = None


def require_api_key() -> None:
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "OPENAI_API_KEY is not configured. Create a .env file "
                "in the backend folder and add OPENAI_API_KEY=your_api_key_here."
            ),
        )


def require_vectorstore() -> Chroma:
    global vectorstore
    if vectorstore is None:
        try:
            vectorstore = load_vectorstore()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Failed to load Chroma vector database. "
                    "Ensure the persistent path is ./chroma_db."
                ),
            ) from exc
    return vectorstore


def create_llm(temperature: float = 0.0) -> ChatOpenAI:
    require_api_key()
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )


# =========================================================
# PROMPTS & CHAINS
# =========================================================

# Preserve the original system prompt and its answer rules.
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful QA assistant that answers questions using the provided document context.

Rules:
1. Use facts from the provided context to answer the user's question.
2. If the document mentions the topic but does not provide a full definition, explain what the document specifically says about it.
3. If the context contains no relevant information at all, state: "I cannot answer this question based on the provided document."

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


# =========================================================
# HELPER FOR PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_documents(file_path: str, source_name: str) -> list[Document]:
    documents: list[Document] = []

    # First attempt: extract selectable text using pdfplumber.
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": source_name, "page": i},
                    )
                )

    if documents:
        return documents

    # If no selectable text exists, render each page and use OCR.
    try:
        with fitz.open(file_path) as pdf:
            for i, page in enumerate(pdf):
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False,
                )
                image_bytes = pixmap.tobytes("png")
                text = pytesseract.image_to_string(
                    Image.open(BytesIO(image_bytes))
                )
                if text and text.strip():
                    documents.append(
                        Document(
                            page_content=text.strip(),
                            metadata={
                                "source": source_name,
                                "page": i,
                                "ocr": True,
                            },
                        )
                    )
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "This PDF appears to be scanned. Install the Tesseract OCR engine "
            "and ensure tesseract.exe is on PATH, then try again."
        ) from error

    return documents


# =========================================================
# HELPER FUNCTIONS: RETRIEVAL, OVERVIEW, ANSWERS
# =========================================================

def format_documents(docs: list[Document]) -> str:
    formatted_documents = []

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "Unknown source")
        page = metadata.get("page")

        if page is not None:
            source_text = f"Source: {source}, Page: {page + 1}"
        else:
            source_text = f"Source: {source}"

        formatted_documents.append(
            f"Document {index}\n{source_text}\nContent:\n{doc.page_content}"
        )

    return "\n\n".join(formatted_documents)


def retrieve_documents(
    question: str,
    k: int,
    threshold: float,
) -> tuple[list[Document], list[float]]:
    store = require_vectorstore()
    results = store.similarity_search_with_score(question, k=k)

    relevant_docs = []
    scores = []

    for doc, score in results:
        numeric_score = float(score)
        scores.append(numeric_score)
        if numeric_score <= threshold:
            relevant_docs.append(doc)

    return relevant_docs, scores


def is_overview_query(question: str) -> bool:
    overview_keywords = [
        "what is in",
        "what is inside",
        "summarize",
        "summary",
        "overview",
        "tell me about",
        "what does the file",
        "what is the document about",
        "what is in doc",
        "about the document",
        "summary of file",
        "summary of the file",
        "file summary",
        "document summary",
    ]
    q_lower = question.lower()
    return any(keyword in q_lower for keyword in overview_keywords)


def get_stored_overview_docs() -> list[Document]:
    """Directly query Chroma for documents tagged type='overview'."""
    try:
        store = require_vectorstore()
        results = store.get(where={"type": "overview"})
        if results and results.get("documents"):
            overview_docs = []
            metadatas = results.get("metadatas") or []
            for i, text in enumerate(results["documents"]):
                metadata = metadatas[i] if i < len(metadatas) else {}
                overview_docs.append(
                    Document(page_content=text or "", metadata=metadata or {})
                )
            return overview_docs
    except HTTPException:
        raise
    except Exception:
        pass

    return []


def generate_answer(
    question: str,
    k: int = 4,
    threshold: float = 2.0,
    temperature: float = 0.0,
) -> dict[str, Any]:
    docs: list[Document] = []
    scores: list[float] = []

    if is_overview_query(question):
        # Use the saved overview first, with the original similarity fallback.
        docs = get_stored_overview_docs()
        if not docs:
            docs, scores = retrieve_documents(question, k, threshold)
        else:
            scores = [0.0] * len(docs)
    else:
        docs, scores = retrieve_documents(question, k, threshold)

    if docs:
        context = format_documents(docs)
        chain = RAG_PROMPT | create_llm(temperature) | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return {
            "answer": answer,
            "source_type": "📚 RAG — Uploaded Documents",
            "documents": docs,
            "scores": scores,
        }

    return {
        "answer": "I cannot answer this question based on the provided document.",
        "source_type": "⚠️ No Relevant Context Found",
        "documents": [],
        "scores": scores,
    }


def document_to_response(doc: Document) -> dict[str, Any]:
    metadata = doc.metadata or {}
    page = metadata.get("page")
    return {
        "source": metadata.get("source", "Unknown source"),
        "page": page + 1 if page is not None else None,
        "content": doc.page_content,
        "metadata": metadata,
    }


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)
    relevance_threshold: float = Field(default=2.0, ge=0.1, le=3.0)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)


# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/")
def root():
    return {"message": "Document Question with RAG API is running."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_api_key_configured": bool(OPENAI_API_KEY),
        "vectorstore_loaded": vectorstore is not None,
    }


@app.get("/api/settings")
def get_settings():
    """Return the same default RAG settings exposed by the Streamlit sidebar."""
    return {
        "temperature": 0.0,
        "top_k": 4,
        "relevance_threshold": 2.0,
        "minimum_top_k": 1,
        "maximum_top_k": 10,
        "minimum_temperature": 0.0,
        "maximum_temperature": 1.0,
        "minimum_relevance_threshold": 0.1,
        "maximum_relevance_threshold": 3.0,
    }


@app.post("/api/documents")
async def process_document(file: UploadFile = File(...)):
    """
    Equivalent of 'Process and Add to Database':
    extract text/OCR, split chunks, replace the previous collection,
    index chunks, and save a generated overview.
    """
    global vectorstore

    require_api_key()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")

    if len(content) > 30 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="The uploaded PDF must be 30 MB or smaller.",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        documents = extract_pdf_documents(tmp_path, file.filename)

        if not documents:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable text found in the PDF. "
                    "If this is a scanned PDF, OCR is required."
                ),
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text found in the PDF.",
            )

        # Preserve the original behaviour: clear the previous collection
        # when a new PDF is processed.
        store = require_vectorstore()
        try:
            store.delete_collection()
        except Exception:
            pass

        vectorstore = load_vectorstore()
        vectorstore.add_documents(chunks)

        sample_content = "\n\n".join(
            [doc.page_content for doc in documents[:5]]
        )
        summary_prompt = (
            "Provide a comprehensive overview of what this document contains, "
            "including its main topics, target subject, and key themes based on "
            "these initial pages:\n\n"
            f"{sample_content}"
        )

        overview_text = create_llm(0.0).invoke(summary_prompt).content

        summary_document = Document(
            page_content=(
                "DOCUMENT OVERVIEW AND SUMMARY:\n"
                f"File Name: {file.filename}\n\n"
                f"What this document is about:\n{overview_text}"
            ),
            metadata={
                "source": file.filename,
                "type": "overview",
            },
        )
        vectorstore.add_documents([summary_document])

        return {
            "message": (
                f"Successfully processed {len(chunks)} chunks + "
                "added document overview!"
            ),
            "filename": file.filename,
            "pages_with_text": len(documents),
            "chunks": len(chunks),
            "overview": overview_text,
            "is_file_processed": True,
        }

    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {exc}",
        ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        await file.close()


@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Equivalent of the Streamlit chat flow.
    The browser/frontend keeps chat history; this endpoint returns the answer,
    source type, retrieved documents, distance scores, and threshold details.
    """
    require_api_key()

    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    try:
        result = generate_answer(
            question=question,
            k=request.top_k,
            threshold=request.relevance_threshold,
            temperature=request.temperature,
        )

        documents = result["documents"]
        return {
            "answer": result["answer"],
            "source_type": result["source_type"],
            "documents": [document_to_response(doc) for doc in documents],
            "scores": result["scores"],
            "relevance_threshold": request.relevance_threshold,
            "passed_filtering": len(documents),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating the answer: {exc}",
        ) from exc


@app.delete("/api/database")
def clear_database():
    """Equivalent of the Streamlit 'Clean Database' button."""
    global vectorstore

    try:
        store = require_vectorstore()
        store.delete_collection()
        vectorstore = load_vectorstore()
        return {
            "message": "Vector database cleared completely!",
            "is_file_processed": False,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error cleaning database: {exc}",
        ) from exc
