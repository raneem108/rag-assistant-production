from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pathlib import Path
import os

# ── Configuration ─────────────────────────────────────────────
DOCS_DIR    = Path("data/documents")
CHROMA_DIR  = Path("chroma_db")
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

def load_documents():
    """
    Load all PDF files from the documents folder.
    PyPDFLoader reads each page as a separate document.
    """
    documents = []
    pdf_files = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DOCS_DIR}")

    print(f"Found {len(pdf_files)} PDF files:")
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        documents.extend(docs)
        print(f"    → {len(docs)} pages loaded")

    print(f"\nTotal pages loaded: {len(documents)}")
    return documents

def split_documents(documents):
    """
    Split documents into smaller chunks.

    Why do we split?
    LLMs have a context window limit — we can't send
    an entire 100-page PDF. We split into chunks of
    500 characters with 50 character overlap.

    The overlap prevents losing context at chunk boundaries.
    Example: if a sentence spans two chunks, the overlap
    ensures both chunks contain part of that sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    return chunks

def create_vectorstore(chunks):
    """
    Convert chunks to embeddings and store in ChromaDB.

    Embeddings = numerical representations of text.
    Similar text → similar numbers → close in vector space.

    We use a multilingual model that handles Arabic and English.
    """
    print(f"\nCreating embeddings with {EMBED_MODEL}...")
    print("This may take a few minutes on first run...")

    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBED_MODEL
    )

    # Create or overwrite the vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    print(f"Vector store created at {CHROMA_DIR}")
    print(f"Total vectors stored: {vectorstore._collection.count()}")
    return vectorstore

def run_ingestion():
    """Main ingestion pipeline — runs once to index all documents."""
    print("=" * 50)
    print("Starting document ingestion pipeline")
    print("=" * 50)

    # Step 1 — Load
    documents = load_documents()

    # Step 2 — Split
    chunks = split_documents(documents)

    # Step 3 — Embed and store
    vectorstore = create_vectorstore(chunks)

    print("\n✅ Ingestion complete!")
    print(f"Documents are ready for querying.")
    return vectorstore

if __name__ == "__main__":
    run_ingestion()