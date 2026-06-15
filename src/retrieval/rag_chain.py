from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# ── Configuration ─────────────────────────────────────────────
CHROMA_DIR  = "chroma_db"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL  = "llama-3.1-8b-instant"

# ── Prompt Template ───────────────────────────────────────────
# This is the instruction we give the LLM.
# {context} = the retrieved document chunks
# {question} = the user's question
PROMPT_TEMPLATE = """
You are a legal assistant specializing in Jordanian data protection law.
Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I cannot find this information 
in the provided documents."
Always cite which document your answer comes from.
Answer in the same language as the question.

Context:
{context}

Question: {question}

Answer:"""

def get_vectorstore():
    """Load the existing ChromaDB vector store."""
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

def format_docs(docs):
    """
    Format retrieved documents into a single context string.
    We include the source filename so the LLM can cite it.
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content
        formatted.append(f"[Source: {source}]\n{content}")
    return "\n\n---\n\n".join(formatted)

def build_rag_chain():
    """
    Build the RAG chain using LangChain's pipe operator.
    
    The chain works like this:
    question → retriever finds top 3 chunks
            → chunks formatted as context
            → prompt template filled in
            → sent to Groq LLM
            → answer extracted as string
    """
    vectorstore = get_vectorstore()
    
    # Retriever — finds top 3 most similar chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # LLM — Groq with Llama3
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=GROQ_MODEL,
        temperature=0,  # 0 = deterministic, no creativity
                        # We want factual legal answers
    )
    
    # Prompt
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Build the chain using | pipe operator
    # This is LangChain Expression Language (LCEL)
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def ask(question: str) -> dict:
    """
    Ask a question and return the answer with sources.
    This is the main function called by the API.
    """
    chain = build_rag_chain()
    vectorstore = get_vectorstore()
    
    # Get answer
    answer = chain.invoke(question)
    
    # Get source documents for transparency
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    source_docs = retriever.invoke(question)
    sources = list(set([
        doc.metadata.get('source', 'Unknown') 
        for doc in source_docs
    ]))
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":
    # Test the chain
    print("Testing RAG chain...")
    print("-" * 50)
    
    test_questions = [
        "What are the rights of data subjects under Jordanian law?",
        "What is a Data Protection Impact Assessment?",
        "ما هي حقوق أصحاب البيانات؟"  # Arabic question
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        result = ask(question)
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 50)