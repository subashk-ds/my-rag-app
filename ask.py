import os
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print("🔍 Loading your document database...")
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorstore = Chroma(persist_directory="./my_document_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatCohere(model="command-r-plus-08-2024")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the following context.
For each fact you state, cite which document chunk it came from like [1], [2], etc.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (with citations):""")

def format_docs(docs):
    return "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n" + "="*50)
print("🤖 ASK ME ANYTHING ABOUT THE DOCUMENT!")
print("="*50)

while True:
    question = input("\n❓ Your question (or type 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    
    print("\n🔎 Searching and generating answer...")
    answer = rag_chain.invoke(question)
    print(f"\n💡 ANSWER:\n{answer}")
    print("\n" + "-"*40)