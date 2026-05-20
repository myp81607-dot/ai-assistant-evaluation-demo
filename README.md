# ai-assistant-evaluation-demo
A lightweight demo for testing RAG-style AI assistant responses, including knowledge base retrieval, test questions, response evaluation, and bad case analysis.
# Mini RAG / AI Assistant Evaluation Demo

This is a lightweight demo for testing a RAG-style AI assistant workflow.  
The project simulates how an AI assistant retrieves knowledge, answers test questions, and records bad cases for response evaluation.

## Project Goal

This project is not a production-level RAG system.  
It is designed to demonstrate my basic understanding of:

- RAG workflow
- Knowledge base testing
- AI assistant response evaluation
- Bad case analysis
- Prompt and answer quality control

## Workflow

1. Prepare a small knowledge base
2. Prepare test questions
3. Retrieve the most relevant knowledge chunk
4. Generate a simple grounded answer
5. Save evaluation results
6. Record bad cases and improvement suggestions

## Repository Structure

```text
ai-assistant-evaluation-demo/
├── README.md
├── requirements.txt
├── src/
│   └── mini_rag_demo.py
├── data/
│   ├── knowledge_base.csv
│   ├── test_questions.csv
│   └── bad_case_table.csv
└── outputs/
    └── evaluation_results.csv

pip install -r requirements.txt
python src/mini_rag_demo.py
