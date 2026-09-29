
"use client";

import { useRef, useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
};

const API_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [documentId, setDocumentId] = useState("");
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [question, setQuestion] = useState("");
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);

  async function uploadPDF() {
    if (!file) {
      setError("Please select a PDF file first.");
      return;
    }

    setUploading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/api/documents`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "PDF upload failed.");
      }

      setDocumentId(data.document_id);
      setMessages([
        {
          role: "assistant",
          content: `Your PDF "${file.name}" has been processed. You can now ask questions about its contents.`,
        },
      ]);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not connect to the Python API."
      );
    } finally {
      setUploading(false);
    }
  }

  async function sendQuestion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const prompt = question.trim();
    if (!prompt || !documentId || sending) return;

    setMessages((previous) => [
      ...previous,
      { role: "user", content: prompt },
    ]);
    setQuestion("");
    setSending(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: documentId,
          question: prompt,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Unable to generate an answer.");
      }

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "The request failed."
      );
    } finally {
      setSending(false);
    }
  }

  function resetDocument() {
    setFile(null);
    setDocumentId("");
    setMessages([]);
    setError("");
    if (fileInput.current) fileInput.current.value = "";
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-icon">R</span>
          <span>Resolve<span className="brand-light">AI</span></span>
        </a>
        <span className="topbar-label">
          <span className="status-dot" /> AI DOCUMENT ASSISTANT
        </span>
      </header>

      <section className="hero">
        <div className="eyebrow">
          <span>✦</span> YOUR DOCUMENTS, UNDERSTOOD
        </div>
        <h1>
          Ask your documents.
          <br />
          <span>Get meaningful answers.</span>
        </h1>
        <p className="hero-description">
          Upload a PDF, explore its contents, and get answers grounded
          in the information inside your document.
        </p>
      </section>

      <section className="workspace">
        <aside className="sidebar">
          <div className="section-heading">
            <span className="step-number">01</span>
            <div>
              <h2>Your document</h2>
              <p>Upload a PDF to begin</p>
            </div>
          </div>

          <button
            className={`upload-zone ${file ? "has-file" : ""}`}
            onClick={() => fileInput.current?.click()}
            type="button"
          >
            <span className="upload-icon">↑</span>
            <strong>{file ? file.name : "Choose a PDF file"}</strong>
            <span>
              {file
                ? "Click to select a different file"
                : "Click here to browse your files"}
            </span>
            <small>PDF FORMAT</small>
          </button>

          <input
            ref={fileInput}
            type="file"
            accept=".pdf,application/pdf"
            className="hidden-input"
            onChange={(event) => {
              const selected = event.target.files?.[0] || null;
              setFile(selected);
              setDocumentId("");
              setMessages([]);
              setError(
                selected && selected.type !== "application/pdf"
                  ? "Please choose a PDF file."
                  : ""
              );
            }}
          />

          <button
            className="primary-button"
            onClick={uploadPDF}
            disabled={!file || uploading || !!documentId}
            type="button"
          >
            {uploading ? "Processing PDF..." : documentId ? "PDF processed ✓" : "Upload & process PDF →"}
          </button>

          {documentId && (
            <button className="text-button" onClick={resetDocument} type="button">
              ↻ Choose another document
            </button>
          )}

          <div className="privacy-note">
            <span>♧</span>
            <p>
              Your questions are answered using document context retrieved
              by the AI system.
            </p>
          </div>

          <div className="sidebar-footer">
            <span className="status-dot" />
            <span>{documentId ? "Document ready" : "Waiting for a document"}</span>
          </div>
        </aside>

        <section className="chat-panel">
          <div className="chat-header">
            <div className="chat-avatar">✦</div>
            <div>
              <h2>Document Assistant</h2>
              <p>AI-powered question answering</p>
            </div>
            <span className="online-badge">● AI</span>
          </div>

          <div className="chat-body">
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-illustration">
                  <div className="paper-icon">▤</div>
                  <span className="sparkle sparkle-one">✦</span>
                  <span className="sparkle sparkle-two">✧</span>
                </div>
                <h3>Let’s explore your PDF</h3>
                <p>
                  Upload a document on the left. Once it has been processed,
                  ask a question to get started.
                </p>
                <div className="suggestion-list">
                  <div>✧ Summarize the main ideas</div>
                  <div>✧ Find important facts and figures</div>
                  <div>✧ Explain a topic in simple words</div>
                </div>
              </div>
            ) : (
              <div className="messages">
                {messages.map((message, index) => (
                  <article className={`message ${message.role}`} key={index}>
                    <div className="message-avatar">
                      {message.role === "user" ? "You" : "✦"}
                    </div>
                    <div className="message-content">
                      <span className="message-author">
                        {message.role === "user" ? "You" : "Assistant"}
                      </span>
                      <p>{message.content}</p>
                      {message.sources && message.sources.length > 0 && (
                        <div className="sources">
                          <strong>Sources</strong>
                          {message.sources.map((source, sourceIndex) => (
                            <div key={sourceIndex}>↳ {source}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  </article>
                ))}
                {sending && (
                  <div className="typing-indicator">
                    <span />
                    <span />
                    <span />
                    <small>Finding an answer...</small>
                  </div>
                )}
              </div>
            )}
          </div>

          {error && <div className="error-message">{error}</div>}

          <form className="composer" onSubmit={sendQuestion}>
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={
                documentId
                  ? "Ask anything about your document..."
                  : "Upload and process a PDF to start asking..."
              }
              disabled={!documentId || sending}
            />
            <button
              type="submit"
              disabled={!documentId || !question.trim() || sending}
              aria-label="Send question"
            >
              ↑
            </button>
          </form>
          <p className="composer-note">
            AI-generated responses can contain mistakes. Verify important details.
          </p>
        </section>
      </section>

      <footer className="page-footer">
        <span>RESOLVE AI</span>
        <span>Retrieval-Augmented Generation · PDF Question Answering</span>
      </footer>
    </main>
  );
}