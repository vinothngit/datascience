from langchain.chains import create_retrieval_chain
from retriver_chains import retrieval_chain, llm, retriever, document_chain
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset 
from bert_score import score

#from evaluvator import evaluate_answer
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def evaluate_answer(generated_answer, gold_answer, context):
    """Evaluate the generated answer using RAGAs."""
    # Prepare the evaluation dataset
    eval_data = {
        "question": [context["input"]],
        "context": [context["context"]],
        "ground_truth": [gold_answer],
        'answer': [generated_answer]
    }
    
    # Perform the evaluation
    dataset = Dataset.from_dict(eval_data)

    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings
    )
    
    return result

# Function to evaluate the generated answer using BERTScore
def evaluateBERT(generated_answer, gold_answer):
    """Evaluate the generated answer using BERTScore."""
    # Compute BERTScore precision, recall, and F1
    P, R, F1 = score([generated_answer], [gold_answer], lang="en", verbose=True)
    
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

# The Question and Answer bot 
while True:
    # Ask for user query 
    user_input = input("\n\n Enter your query (type 'exit' to quit): ")
    
    if user_input.lower() == "exit":
        print("\nExiting the Q&A bot. Goodbye!")
        break

    query = {"input": user_input} 

    # Use the input query in the invoke function
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    response = retrieval_chain.invoke(query)

    # Print the generated answer
    generated_answer = response['answer']
    print("\n\n Generated Answer: ", response['answer'])

     # Ask if the user wants to assess the generated answer
    assess_answer = input("\n Would you like to assess the generated answer? (yes/no): ").strip().lower()

    if assess_answer == "yes":
        # Ask for the gold standard answer
        gold_answer = input("\n Please provide the gold standard answer for this query: ").strip()

     # Use the context from the response for evaluation
        context = {
            "input": user_input,
            "context": response['context']  # Assuming the context is available in the response
        }
        
        
         # Evaluate the answer using BERTScore
        bert_score_result = evaluateBERT(generated_answer, gold_answer)
        
        # Print BERTScore evaluation results
        print("\n\nEvaluation Results (BERTScore):\n")
        print(f"Precision: {bert_score_result['precision']:.4f}")
        print(f"Recall: {bert_score_result['recall']:.4f}")
        print(f"F1 Score: {bert_score_result['f1']:.4f}")



        # Evaluate the answer using RAGAs
        
        evaluation_result = evaluate_answer(response, gold_answer, context)
        
        # Print evaluation results
        print("\n\nEvaluation Results:\n")
        print(f"Context Precision: {evaluation_result['context_precision']:.4f}")
        print(f"Context Recall: {evaluation_result['context_recall']:.4f}")
        print(f"Faithfulness: {evaluation_result['faithfulness']:.4f}")
        print(f"Answer Relevancy: {evaluation_result['answer_relevancy']:.4f}")
        
    # Proceed to the next query
    print("\n\nProceeding to the next query...\n")

    # Go back to asking for user input for the next question