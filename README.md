# RAG Application for Question Answering

## What is this project?
This is a complete RAG (Retrieval-Augmented Generation) system I built from scratch. It can answer questions based on your own documents. Think of it like a ChatGPT that you can teach your own files.

## What problem does it solve?
Normal AI like ChatGPT only knows public information. This RAG system can read your private documents (`my_document.txt`) and answer questions using ONLY that information. It is useful for company knowledge bases, personal study assistants, or customer support.

## What AI Skills Does This Project Show?
- **Python** (all code is in Python)
- **RAG Architecture** (retrieval + generation)
- **NLP** (text processing, chunking, embedding)
- **Vector Search** (finding relevant document pieces)
- **LLM Integration** (using an LLM to generate answers)

## How to Run This Project
1.  Clone the repository
2.  Install requirements: `pip install -r requirements.txt`
3.  Add your own API key in a `.env` file (I will show you how)
4.  Run `python ask.py "your question here"`

## Files in This Project
| File | What It Does |
| :--- | :--- |
| `ask.py` | Main script to ask questions to the RAG system |
| `prepare_docs.py` | Loads and splits your document into chunks |
| `hybrid_rag.py` | Combines different search methods for better answers |
| `rerank_rag.py` | Improves results by re-ranking the best answers |
| `evaluate.py` | Tests how accurate the system is |

## What I Learned
Building this project taught me how to:
- Connect different AI components (embedding, vector store, LLM)
- Handle document loading and text splitting
- Use environment variables for security (my latest update)
- Structure a multi-file Python project

## My LinkedIn / Contact
[Add your LinkedIn profile link here]
