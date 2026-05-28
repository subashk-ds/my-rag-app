import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

print("📂 Loading document...")
loader = TextLoader("my_document.txt")
documents = loader.load()
print(f"✅ Loaded {len(documents)} document")

print("✂️ Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

print("🧠 Creating embeddings...")
embeddings = CohereEmbeddings(model="embed-english-v3.0")

print("💾 Storing in database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./my_document_db"
)

print("🎉 SUCCESS! Your document is ready!")