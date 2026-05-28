import os
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

print("="*70)
print("🚀 PROJECT 3: RAG WITH CROSS-ENCODER RERANKING")
print("="*70)

# Step 1: Load and prepare document
print("\n📂 Step 1: Loading your document...")
loader = TextLoader("my_document.txt")
documents = loader.load()

print("✂️ Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# Step 2: Create vector database
print("\n🧠 Step 3: Creating vector embeddings...")
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./rerank_db"
)
print("✅ Vector database ready")

# Step 3: Create BM25 index
print("\n🔤 Step 4: Creating BM25 keyword index...")
all_chunks_text = [chunk.page_content for chunk in chunks]
tokenized_chunks = [text.split() for text in all_chunks_text]
bm25 = BM25Okapi(tokenized_chunks)
print("✅ BM25 index ready")

# Step 4: Load Cross-Encoder for reranking (NEW for Project 3!)
print("\n🔄 Step 5: Loading Cross-Encoder reranker...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("✅ Reranker ready!")

# Step 5: Hybrid search WITHOUT reranking (for comparison)
def hybrid_search_without_rerank(query, k=10):
    """Get initial results using hybrid search"""
    # Vector search
    vector_results = vectorstore.similarity_search(query, k=k)
    vector_texts = [doc.page_content for doc in vector_results]
    
    # BM25 search
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:k]
    bm25_texts = [all_chunks_text[i] for i in bm25_indices]
    
    # Combine
    combined = list(dict.fromkeys(vector_texts + bm25_texts))
    return combined[:k]

# Step 6: Hybrid search WITH reranking (NEW for Project 3!)
def hybrid_search_with_rerank(query, k=5):
    """Get initial results, then rerank to find the BEST ones"""
    # First, get more results (k*2) before reranking
    initial_results = hybrid_search_without_rerank(query, k=10)
    
    print(f"   📊 Found {len(initial_results)} candidates, reranking to find best {k}...")
    
    # Create pairs for reranker (query + each document)
    pairs = [[query, doc] for doc in initial_results]
    
    # Get relevance scores from reranker
    scores = reranker.predict(pairs)
    
    # Combine documents with scores
    scored_docs = list(zip(initial_results, scores))
    
    # Sort by score (higher is better)
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k documents
    return [doc for doc, score in scored_docs[:k]]

# Step 7: RAG with citations
print("\n🤖 Step 6: Initializing AI model...")
llm = ChatCohere(model="command-r-plus-08-2024")

prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context.
Cite sources using [1], [2], etc.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")

def ask_with_rerank(question):
    print(f"\n🔍 Question: '{question}'")
    print("-" * 50)
    
    # Get results WITH reranking
    print("🔄 Using RERANKED search...")
    relevant_docs = hybrid_search_with_rerank(question)
    context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(relevant_docs)])
    
    response = llm.invoke(prompt.format(context=context, question=question))
    
    print(f"\n💡 ANSWER:\n{response.content}")
    print("\n📚 BEST SOURCES (after reranking):")
    for i, doc in enumerate(relevant_docs):
        print(f"   [{i+1}] {doc[:80]}...")

# Step 8: Interactive chat
print("\n" + "="*70)
print("💬 PROJECT 3 READY! Cross-Encoder Reranking is ACTIVE")
print="="*70
print("\n🎯 Reranking finds the BEST sources before answering!")
print("="*70)

while True:
    question = input("\n❓ Your question (or 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    ask_with_rerank(question)

print("\n🎉 Congratulations! You've completed Project 3 - RAG with Reranking!")
print("   Your system now finds the MOST relevant information!")