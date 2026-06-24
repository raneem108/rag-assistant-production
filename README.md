\# Jordan Legal RAG Assistant



A production RAG (Retrieval-Augmented Generation) system that answers questions about Jordanian data protection law in both Arabic and English, with source citations from official legal documents.



\##  Live Demo

\[Try it →](https://jordan-legal-rag-raneem.streamlit.app)



\## 📖 What This Project Does



Lawyers, compliance officers, and businesses need to search through dense legal text to answer questions like:

\- What are the rights of data subjects under Jordanian law?

\- When is a Data Protection Impact Assessment required?

\- What are the penalties for a data breach?



This assistant retrieves the exact relevant clauses from official Jordanian legal documents and generates accurate, cited answers — in the same language the question was asked.



\## Architecture
PDF Documents (Jordanian Law)



↓



Document Loader + Text Splitter



↓



Multilingual Embeddings (Arabic + English)



↓



ChromaDB Vector Store



↓



User Question → Similarity Search (top 3 chunks)



↓



Groq LLaMA 3.1 (temperature=0)



↓



Cited Answer (matched to question's language)



\## 🛠️ Tech Stack



| Layer | Technology |

|---|---|

| LLM | Groq LLaMA 3.1 8B Instant |

| Embeddings | sentence-transformers (multilingual) |

| Vector Store | ChromaDB |

| Orchestration | LangChain (LCEL) |

| Backend | FastAPI |

| Frontend | Streamlit |

| Language | Python 3.14 |



\##  Project Structure

rag-assistant-production/



├── data/



│   └── documents/         # Official Jordanian legal PDFs



├── src/



│   ├── ingestion/



│   │   └── ingest.py      # PDF → chunks → embeddings → ChromaDB



│   ├── retrieval/



│   │   └── rag\_chain.py   # RAG pipeline (LCEL chain)



│   └── api/



│       └── main.py        # FastAPI REST endpoints



├── app.py                 # Streamlit chat UI



└── requirements.txt



\## Source Documents



\- Personal Data Protection Law No. 24 of 2023 (Official Gazette)

\- Data Protection Impact Assessment (DPIA) Guidelines

\- Data Disclosure Regulation 2025

\- Official English translation of the Data Protection Law



\##  Key Engineering Decisions



\*\*Multilingual embeddings\*\* — Used `paraphrase-multilingual-MiniLM-L12-v2` so the system retrieves relevant content regardless of whether the source document or question is in Arabic or English.



\*\*temperature=0\*\* — Legal answers must be deterministic and consistent, not creative. Same question always produces the same answer.



\*\*Strict context-only prompting\*\* — The LLM is instructed to answer only from retrieved document chunks and explicitly say when information isn't available, preventing hallucination — critical for legal accuracy.



\*\*Language-matching enforcement\*\* — Initially the system mixed Arabic source text into English answers. Fixed by adding an explicit instruction requiring full translation and forbidding language mixing within a single response.



\##  Running Locally



```bash

git clone https://github.com/raneem108/rag-assistant-production.git

cd rag-assistant-production



python -m venv venv

venv\\Scripts\\activate



pip install -r requirements.txt



\# Add GROQ\_API\_KEY to .env



\# Build the vector store (first run only)

python src/ingestion/ingest.py



\# Run the API

uvicorn src.api.main:app --reload --port 8000



\# In a separate terminal, run the chat UI

streamlit run app.py

```



\##  Example Queries



\- "What are the rights of data subjects under Jordanian law?"

\- "When is a DPIA required?"

\- "ما هي حقوق أصحاب البيانات؟"

\- "متى يجب إجراء تقييم أثر حماية البيانات؟"



\##  Author



\*\*Raneem Abujabal\*\* — Data Science \& AI Graduate, Yarmouk University

\- GitHub: \[raneem108](https://github.com/raneem108)

\- Email: rabujabal4@gmail.com

