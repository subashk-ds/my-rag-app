import os
import json
from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

load_dotenv()

print("="*70)
print("📊 PROJECT 4: RAGAS EVALUATION PIPELINE")
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
    {
        "question": "What is my name?",
        "ground_truth": "Rahul"
    },
    {
        "question": "Where do I live?",
        "ground_truth": "Mumbai, India"
    },
    {
        "question": "What is my favorite food?",
        "ground_truth": "Pizza, margherita pizza with extra cheese"
    },
    {
        "question": "What do I want to build?",
        "ground_truth": "A RAG application that can answer questions from documents"
    },
    {
        "question": "How many hours do I practice coding?",
        "ground_truth": "2 hours every day"
    },
    {
        "question": "What is my favorite subject?",
        "ground_truth": "Python programming"
    }
]

print(f"✅ Created {len(test_questions)} test questions")

# Step 3: Run RAG system on all test questions
print("\n🔍 Step 3: Running RAG system on test questions...")

answers = []
contexts = []

for i, item in enumerate(test_questions, 1):
    print(f"   Testing {i}/{len(test_questions)}: '{item['question']}'...")
    answer, context = get_answer(item['question'])
    answers.append(answer)
    contexts.append(context)

print("✅ All tests completed!")

# Step 4: Create dataset for RAGAS evaluation
print("\n📊 Step 4: Running RAGAS evaluation...")

dataset = Dataset.from_dict({
    "question": [q["question"] for q in test_questions],
    "answer": answers,
    "contexts": contexts,
    "ground_truth": [q["ground_truth"] for q in test_questions]
})

# Step 5: Calculate scores
scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)

# Step 6: Display results
print("\n" + "="*70)
print("📈 EVALUATION RESULTS")
print("="*70)

print(f"\n📊 Faithfulness: {scores['faithfulness']:.2f}")
print("   (Does the answer match the sources?)")

print(f"\n📊 Answer Relevancy: {scores['answer_relevancy']:.2f}")
print("   (Is the answer related to the question?)")

print(f"\n📊 Context Precision: {scores['context_precision']:.2f}")
print("   (Are the sources correctly used?)")

# Step 7: Overall score
overall_score = (scores['faithfulness'] + scores['answer_relevancy'] + scores['context_precision']) / 3
print("\n" + "="*70)
print(f"🏆 OVERALL SCORE: {overall_score:.2f} / 1.00")
print("="*70)

if overall_score >= 0.7:
    print("\n🎉 EXCELLENT! Your RAG system is production-ready!")
elif overall_score >= 0.5:
    print("\n👍 GOOD! Your RAG system works well, with room for improvement.")
else:
    print("\n⚠️ NEEDS IMPROVEMENT! Try adjusting chunk size or adding more documents.")

# Step 8: Save results
results = {
    "faithfulness": float(scores['faithfulness']),
    "answer_relevancy": float(scores['answer_relevancy']),
    "context_precision": float(scores['context_precision']),
    "overall_score": float(overall_score),
    "total_questions": len(test_questions)
}

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n💾 Results saved to 'evaluation_results.json'")
print("\n🎯 Project 4 Complete! You can now MEASURE your RAG quality!")