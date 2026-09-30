
import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import chromadb
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


# Preserve the Windows Tesseract fallback for local development.
if not shutil.which("tesseract"):
    default_tesseract_path = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if os.path.isfile(default_tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = (
            default_tesseract_path
        )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Document Question with RAG",
    version="1.0.0",
    description="PDF question answering using OpenAI and Chroma Cloud.",
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

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
# CHROMA CLOUD VECTOR DATABASE
# =========================================================

# IMPORTANT:
# Do not create BASE_DIR / "chroma_db".
# Vercel's deployment directory is read-only.
# This application uses Chroma Cloud for persistent storage.


def load_vectorstore() -> Chroma:
    """Connect to the persistent Chroma Cloud database."""

    required_variables = [
        "CHROMA_TENANT",
        "CHROMA_DATABASE",
        "CHROMA_API_KEY",
    ]

    missing_variables = [
        name
        for name in required_variables
        if not os.getenv(name)
    ]

    if missing_variables:
        raise RuntimeError(
            "Missing Chroma Cloud environment variables: "
            + ", ".join(missing_variables)
        )

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is missing."
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )

    client = chromadb.CloudClient(
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
        api_key=os.environ["CHROMA_API_KEY"],
    )

    return Chroma(
        client=client,
        collection_name="pdf_documents",
        embedding_function=embeddings,
    )


# =========================================================
# INITIALIZE VECTOR DATABASE
# =========================================================

vectorstore = None
VECTORSTORE_STARTUP_ERROR = None

try:
    vectorstore = load_vectorstore()

except Exception as exc:
    # Keep the API importable if Chroma Cloud is unavailable.
    # /docs and /openapi.json should still be accessible.
    VECTORSTORE_STARTUP_ERROR = str(exc)

    print(
        "Chroma Cloud initialization failed:",
        VECTORSTORE_STARTUP_ERROR,
    )


def require_api_key() -> None:
    """Ensure the OpenAI API key is configured."""

    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured. "
            "Set it in Vercel Environment Variables "
            "or your local backend .env file.",
        )


def require_vectorstore() -> Chroma:
    """Return the Chroma Cloud connection or retry initialization."""

    global vectorstore, VECTORSTORE_STARTUP_ERROR

    if vectorstore is None:
        try:
            vectorstore = load_vectorstore()
            VECTORSTORE_STARTUP_ERROR = None

        except Exception as exc:
            VECTORSTORE_STARTUP_ERROR = str(exc)

            print(
                "Chroma Cloud connection failed:",
                VECTORSTORE_STARTUP_ERROR,
            )

            raise HTTPException(
                status_code=503,
                detail="Unable to connect to Chroma Cloud. "
                "Check your environment variables and "
                "database availability.",
            ) from exc

    return vectorstore


def create_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Create the OpenAI chat model."""

    require_api_key()

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=OPENAI_API_KEY,
    )


# =========================================================
# PROMPTS AND CHAINS
# =========================================================

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful QA assistant that answers
questions using the provided document context.

Rules:

1. Use facts from the provided context to answer the user's question.
2. If the document mentions the topic but does not provide a
   full definition, explain what the document specifically says.
3. If the context contains no relevant information at all,
   state: "I cannot answer this question based on the provided document."

Context:

{context}""",
        ),
        ("human", "{question}"),
    ]
)


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_documents(
    file_path: str,
    source_name: str,
) -> list[Document]:
    """Extract selectable PDF text, then try OCR for scanned PDFs."""

    documents: list[Document] = []

    # First attempt: extract selectable text.
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source_name,
                            "page": i,
                        },
                    )
                )

    if documents:
        return documents

    # Second attempt: OCR for scanned PDFs.
    try:
        with fitz.open(file_path) as pdf:
            for i, page in enumerate(pdf):
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False,
                )

                image_bytes = pixmap.tobytes("png")

                with Image.open(BytesIO(image_bytes)) as image:
                    text = pytesseract.image_to_string(image)

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

    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "This PDF appears to be scanned, but the Tesseract "
            "OCR engine is not installed or accessible. "
            "Configure Tesseract in the deployment environment "
            "to process scanned PDFs."
        ) from exc

    return documents


# =========================================================
# HELPER: FORMAT RETRIEVED DOCUMENTS
# =========================================================

def format_documents(docs: list[Document]) -> str:
    """Format retrieved chunks into context for the LLM."""

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
            f"Document {index}\n"
            f"{source_text}\n"
            f"Content:\n{doc.page_content}"
        )

    return "\n\n".join(formatted_documents)


# =========================================================
# HELPER: RETRIEVE DOCUMENTS
# =========================================================

def retrieve_documents(
    question: str,
    k: int,
    threshold: float,
) -> tuple[list[Document], list[float]]:
    """Retrieve relevant chunks from Chroma Cloud."""

    store = require_vectorstore()

    results = store.similarity_search_with_score(
        question,
        k=k,
    )

    relevant_docs = []
    scores = []

    for doc, score in results:
        numeric_score = float(score)
        scores.append(numeric_score)

        if numeric_score <= threshold:
            relevant_docs.append(doc)

    return relevant_docs, scores


# =========================================================
# HELPER: IDENTIFY OVERVIEW QUESTIONS
# =========================================================

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

    return any(
        keyword in q_lower
        for keyword in overview_keywords
    )


# =========================================================
# HELPER: LOAD STORED OVERVIEW
# =========================================================

def get_stored_overview_docs() -> list[Document]:
    """Retrieve documents tagged with type='overview'."""

    try:
        store = require_vectorstore()

        results = store.get(
            where={"type": "overview"}
        )

        if results and results.get("documents"):
            overview_docs = []
            metadatas = results.get("metadatas") or []

            for i, text in enumerate(results["documents"]):
                metadata = (
                    metadatas[i]
                    if i < len(metadatas)
                    else {}
                )

                overview_docs.append(
                    Document(
                        page_content=text or "",
                        metadata=metadata or {},
                    )
                )

            return overview_docs

    except HTTPException:
        raise

    except Exception as exc:
        print("Could not load stored overview:", exc)

    return []


# =========================================================
# HELPER: GENERATE ANSWER
# =========================================================

def generate_answer(
    question: str,
    k: int = 4,
    threshold: float = 2.0,
    temperature: float = 0.0,
) -> dict[str, Any]:

    docs: list[Document] = []
    scores: list[float] = []

    if is_overview_query(question):
        # Use the saved overview first.
        docs = get_stored_overview_docs()

        if not docs:
            docs, scores = retrieve_documents(
                question,
                k,
                threshold,
            )
        else:
            scores = [0.0] * len(docs)

    else:
        docs, scores = retrieve_documents(
            question,
            k,
            threshold,
        )

    if docs:
        context = format_documents(docs)

        chain = (
            RAG_PROMPT
            | create_llm(temperature)
            | StrOutputParser()
        )

        answer = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        return {
            "answer": answer,
            "source_type": "RAG - Uploaded Documents",
            "documents": docs,
            "scores": scores,
        }

    return {
        "answer": (
            "I cannot answer this question based on "
            "the provided document."
        ),
        "source_type": "No Relevant Context Found",
        "documents": [],
        "scores": scores,
    }


# =========================================================
# HELPER: DOCUMENT RESPONSE
# =========================================================

def document_to_response(
    doc: Document,
) -> dict[str, Any]:

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
    relevance_threshold: float = Field(
        default=2.0,
        ge=0.1,
        le=3.0,
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Document Question with RAG API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_api_key_configured": bool(OPENAI_API_KEY),
        "vectorstore_loaded": vectorstore is not None,
        "vectorstore_error": VECTORSTORE_STARTUP_ERROR,
    }


@app.get("/api/settings")
def get_settings():
    """Return the default RAG settings."""

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


# =========================================================
# UPLOAD AND PROCESS PDF
# =========================================================

@app.post("/api/documents")
async def process_document(
    file: UploadFile = File(...),
):
    """
    Extract PDF text, split chunks, replace the existing
    collection, index chunks, and save a document overview.
    """

    global vectorstore, VECTORSTORE_STARTUP_ERROR

    require_api_key()

    if (
        not file.filename
        or not file.filename.lower().endswith(".pdf")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty.",
        )

    if len(content) > 30 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="The uploaded PDF must be 30 MB or smaller.",
        )

    tmp_path = None

    try:
        # Temporary PDF processing uses a writable temp directory.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        documents = extract_pdf_documents(
            tmp_path,
            file.filename,
        )

        if not documents:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable text found in the PDF. "
                    "Scanned PDFs require a working OCR engine."
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

        # Preserve the original replacement behavior.
        store = require_vectorstore()

        store.delete_collection()

        vectorstore = load_vectorstore()
        VECTORSTORE_STARTUP_ERROR = None

        vectorstore.add_documents(chunks)

        sample_content = "\n\n".join(
            doc.page_content
            for doc in documents[:5]
        )

        summary_prompt = (
            "Provide a comprehensive overview of what this "
            "document contains, including its main topics, "
            "target subject, and key themes based on these "
            "initial pages:\n\n"
            f"{sample_content}"
        )

        overview_result = create_llm(0.0).invoke(
            summary_prompt
        )

        overview_text = (
            overview_result.content
            if isinstance(overview_result.content, str)
            else str(overview_result.content)
        )

        summary_document = Document(
            page_content=(
                "DOCUMENT OVERVIEW AND SUMMARY:\n"
                f"File Name: {file.filename}\n\n"
                "What this document is about:\n"
                f"{overview_text}"
            ),
            metadata={
                "source": file.filename,
                "type": "overview",
            },
        )

        vectorstore.add_documents(
            [summary_document]
        )

        return {
            "message": (
                f"Successfully processed {len(chunks)} chunks "
                "and added the document overview."
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
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print("PDF processing failed:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {exc}",
        ) from exc

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        await file.close()


# =========================================================
# CHAT WITH DOCUMENTS
# =========================================================

@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Answer questions using retrieved document context.
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
            "documents": [
                document_to_response(doc)
                for doc in documents
            ],
            "scores": result["scores"],
            "relevance_threshold": (
                request.relevance_threshold
            ),
            "passed_filtering": len(documents),
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("Chat request failed:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while generating the answer."
            ),
        ) from exc


# =========================================================
# CLEAR VECTOR DATABASE
# =========================================================

@app.delete("/api/database")
def clear_database():
    """Clear the Chroma Cloud collection."""

    global vectorstore, VECTORSTORE_STARTUP_ERROR

    try:
        store = require_vectorstore()

        store.delete_collection()

        vectorstore = load_vectorstore()
        VECTORSTORE_STARTUP_ERROR = None

        return {
            "message": "Vector database cleared completely!",
            "is_file_processed": False,
        }

    except Exception as exc:
        print("Database clearing failed:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail="Error clearing the vector database.",
        ) from exc