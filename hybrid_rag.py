import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

load_dotenv()

print("="*60)
print("🚀 PROJECT 2: HYBRID RAG SYSTEM (BM25 + Vector Search)")
print("="*60)

# Step 1: Load and prepare document
print("\n📂 Loading your document...")
loader = TextLoader("my_document.txt")
documents = loader.load()

print("✂️ Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# Step 2: Create vector database
print("\n🧠 Creating vector embeddings...")
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./hybrid_db"
)
print("✅ Vector database ready")

# Step 3: Create BM25 index
print("\n🔤 Creating BM25 keyword index...")
all_chunks_text = [chunk.page_content for chunk in chunks]
tokenized_chunks = [text.split() for text in all_chunks_text]
bm25 = BM25Okapi(tokenized_chunks)
print("✅ BM25 index ready")

# Step 4: Hybrid search function
def hybrid_search(query, k=5):
    # Vector search (finds by meaning)
    vector_results = vectorstore.similarity_search(query, k=k*2)
    vector_texts = [doc.page_content for doc in vector_results]
    
    # BM25 search (finds by exact words)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:k*2]
    bm25_texts = [all_chunks_text[i] for i in bm25_indices]
    
    # Combine and remove duplicates
    combined = list(dict.fromkeys(vector_texts + bm25_texts))
    return combined[:k]

# Step 5: RAG with citations
print("\n🤖 Initializing AI model...")
llm = ChatCohere(model="command-r-plus-08-2024")

prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context.
Cite sources using [1], [2], etc.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")

def ask_hybrid(question):
    print(f"\n🔍 Searching: '{question}'")
    print("   Using HYBRID search (BM25 + Vector)")
    
    relevant_docs = hybrid_search(question)
    context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(relevant_docs)])
    
    response = llm.invoke(prompt.format(context=context, question=question))
    
    print(f"\n💡 ANSWER:\n{response.content}")
    print("\n📚 Sources:")
    for i, doc in enumerate(relevant_docs[:3]):
        print(f"   [{i+1}] {doc[:80]}...")

print("\n" + "="*60)
print("💬 PROJECT 2 READY! Type 'quit' to exit")
print("="*60)

while True:
    question = input("\n❓ Your question: ")
    if question.lower() == 'quit':
        break
    ask_hybrid(question)

print("\n🎉 Project 2 Complete!")