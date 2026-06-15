from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

CHROMA_DIR  = "chroma_db"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

count = vectorstore._collection.count()
print(f"Total vectors in store: {count}")

# Test a sample query
results = vectorstore.similarity_search(
    "What are the rights of data subjects?",
    k=3
)

print(f"\nTop 3 results for test query:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content[:300])
    print(f"Source: {doc.metadata.get('source', 'unknown')}")