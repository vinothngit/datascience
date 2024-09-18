from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from QAapp import gold_answer, context
from bert_score import score

def evaluate_answer(generated_answer, gold_answer, context):
    #Evaluate the generated answer using RAGAs
    # Prepare the evaluation dataset
    eval_data = {
        "question": [context["input"]],
        "context": [context["context"]],
        "ground_truth": [gold_answer],
    }
    
    # Perform the evaluation
    result = evaluate(
        eval_data,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )
    
    return result

# Function to evaluate the generated answer using BERTScore
def evaluateBERT(generated_answer, gold_answer):
    # Compute BERTScore precision, recall, and F1
    P, R, F1 = score([generated_answer], [gold_answer], lang="en", verbose=True)
    
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }