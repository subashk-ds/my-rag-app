import os
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

print("="*70)
print("📊 PROJECT 4: SIMPLE EVALUATION PIPELINE (No C++ Needed)")
print("="*70)

# Step 1: Load your RAG system
print("\n📂 Step 1: Loading RAG system...")

embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorstore = Chroma(persist_directory="./rerank_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatCohere(model="command-r-plus-08-2024")

prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context.
Cite sources using [1], [2], etc.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")

def get_answer(question):
    """Get answer from RAG system"""
    docs = retriever.invoke(question)
    context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
    response = llm.invoke(prompt.format(context=context, question=question))
    return response.content, [doc.page_content for doc in docs]

print("✅ RAG system loaded!")

# Step 2: Create test dataset
print("\n📝 Step 2: Creating test dataset...")

test_questions = [
    {"question": "What is my name?", "expected_keyword": "Rahul"},
    {"question": "Where do I live?", "expected_keyword": "Mumbai"},
    {"question": "What is my favorite food?", "expected_keyword": "pizza"},
    {"question": "What do I want to build?", "expected_keyword": "RAG"},
    {"question": "How many hours do I practice coding?", "expected_keyword": "2 hours"},
    {"question": "What is my favorite subject?", "expected_keyword": "Python"}
]

print(f"✅ Created {len(test_questions)} test questions")

# Step 3: Run RAG system on all test questions
print("\n🔍 Step 3: Running RAG system on test questions...")

results = []

for i, item in enumerate(test_questions, 1):
    print(f"\n   Testing {i}/{len(test_questions)}: '{item['question']}'...")
    answer, contexts = get_answer(item['question'])
    
    # Check if expected keyword is in the answer
    found = item['expected_keyword'].lower() in answer.lower()
    
    results.append({
        "question": item['question'],
        "answer": answer,
        "expected": item['expected_keyword'],
        "passed": found
    })
    
    print(f"   ✅ Answer: {answer[:100]}...")
    print(f"   {'✅ PASSED' if found else '❌ FAILED'} (Expected keyword: '{item['expected_keyword']}')")

# Step 4: Calculate scores
print("\n" + "="*70)
print("📈 EVALUATION RESULTS")
print("="*70)

passed_count = sum(1 for r in results if r["passed"])
score = passed_count / len(results)

print(f"\n📊 Total Questions: {len(results)}")
print(f"✅ Passed: {passed_count}")
print(f"❌ Failed: {len(results) - passed_count}")
print(f"🏆 ACCURACY SCORE: {score * 100:.1f}%")

print("\n📊 DETAILED RESULTS:")
for r in results:
    status = "✅" if r["passed"] else "❌"
    print(f"   {status} {r['question']} → Found '{r['expected']}'? {r['passed']}")

# Step 5: Save results
import json
with open("evaluation_results.json", "w") as f:
    json.dump({
        "score": score,
        "passed": passed_count,
        "total": len(results),
        "details": results
    }, f, indent=2)

print("\n💾 Results saved to 'evaluation_results.json'")

if score >= 0.7:
    print("\n🎉 EXCELLENT! Your RAG system is production-ready!")
elif score >= 0.5:
    print("\n👍 GOOD! Your RAG system works well, with room for improvement.")
else:
    print("\n⚠️ NEEDS IMPROVEMENT! Try adjusting chunk size or adding more documents.")

print("\n🎯 Project 4 Complete! You can now MEASURE your RAG quality!")