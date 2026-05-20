import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    kb = pd.read_csv("data/knowledge_base.csv")
    questions = pd.read_csv("data/test_questions.csv")
    return kb, questions


def build_retriever(kb):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(kb["content"])
    return vectorizer, doc_vectors


def retrieve(question, kb, vectorizer, doc_vectors, top_k=1):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, doc_vectors)[0]
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "doc_id": kb.iloc[idx]["doc_id"],
            "title": kb.iloc[idx]["title"],
            "content": kb.iloc[idx]["content"],
            "score": scores[idx]
        })
    return results


def generate_answer(question, retrieved_docs):
    context = retrieved_docs[0]["content"]
    answer = f"根据知识库资料：{context}"
    return answer


def evaluate_answer(answer, retrieved_docs):
    grounded = "根据知识库资料" in answer
    has_context = retrieved_docs[0]["content"] in answer

    return {
        "grounded": grounded,
        "has_context": has_context,
        "comment": "Demo evaluation only. Manual review is still needed."
    }


def main():
    kb, questions = load_data()
    vectorizer, doc_vectors = build_retriever(kb)

    rows = []

    for _, row in questions.iterrows():
        qid = row["question_id"]
        question = row["question"]

        retrieved_docs = retrieve(question, kb, vectorizer, doc_vectors, top_k=1)
        answer = generate_answer(question, retrieved_docs)
        evaluation = evaluate_answer(answer, retrieved_docs)

        rows.append({
            "question_id": qid,
            "question": question,
            "retrieved_doc_id": retrieved_docs[0]["doc_id"],
            "retrieved_title": retrieved_docs[0]["title"],
            "similarity_score": round(float(retrieved_docs[0]["score"]), 4),
            "answer": answer,
            "grounded": evaluation["grounded"],
            "has_context": evaluation["has_context"],
            "comment": evaluation["comment"]
        })

    output = pd.DataFrame(rows)
    output.to_csv("outputs/evaluation_results.csv", index=False, encoding="utf-8-sig")

    print("Evaluation finished. Results saved to outputs/evaluation_results.csv")


if __name__ == "__main__":
    main()
