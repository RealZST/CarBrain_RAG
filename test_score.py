# coding=utf-8

'''
Online evaluation
'''

import json
import numpy as np
from text2vec import SentenceModel, semantic_search


# Similarity model
model_name = "shibing624/text2vec-base-chinese"
simModel = SentenceModel(model_name_or_path=model_name, device='cuda:0')


# Jaccard similarity calculation
def calc_jaccard(list_a, list_b, threshold=0.3):
    size_b = len(list_b)
    list_c = [i for i in list_a if i in list_b]
    size_c = len(list_c)
    score = size_c / (size_b + 1e-6)
    if score > threshold:
        return 1
    else:
        return 0


# Compute similarity scores between LLM-generated answers and reference answers
def report_score(gold_path, predict_path):
    # Load reference answers
    gold_info = json.load(open(gold_path))
    # Load LLM-generated answers
    pred_info = json.load(open(predict_path))

    idx = 0
    for gold, pred in zip(gold_info, pred_info):
        question = gold["question"]
        keywords = gold["keywords"]
        gold = gold["answer"].strip()
        pred = pred["answer_4"].strip()  # Evaluate only the merged and re-ranked answer

        # If the reference answer is "无答案", the LLM-generated answer must also be "无答案"
        # otherwise, the score is 0
        if gold == "无答案" and pred != gold:
            score = 0.0
        elif gold == "无答案" and pred == gold:
            score = 1.0
        else:
            # Compute semantic similarity using the text2vec model
            semantic_score = semantic_search(
                simModel.encode([gold]),
                simModel.encode(pred),
                top_k=1
            )[0][0]['score']
            # Compute keyword matching score
            join_keywords = [word for word in keywords if word in pred]
            keyword_score = calc_jaccard(join_keywords, keywords)
            # Final score = 50% semantic similarity + 50% keyword matching
            score = 0.5 * keyword_score + 0.5 * semantic_score

        # Store score and LLM-generated answer
        gold_info[idx]["score"] = score
        gold_info[idx]["predict"] = pred
        idx += 1

        print(f"LLM-generated answer: {question}, Score: {score}")

    return gold_info


if __name__ == "__main__":
    '''
    Online evaluation
    '''

    # Path to reference answers
    gold_path = "./data/gold.json"
    print("Read gold from %s" % gold_path)

    # Path to generated answers
    predict_path = "./data/result.json"
    print("Read predict file from %s" % predict_path)

    # Compute scores for each answer
    results = report_score(gold_path, predict_path)

    # Compute the average score
    final_score = np.mean([item["score"] for item in results])
    print("\n")
    print("="*100)
    print(f"Total predictions: {len(results)}, Average prediction score: {final_score}")
    print("="*100)

    # Save the results
    metric_path = "./data/metrics.json"
    results_info = json.dumps(results, ensure_ascii=False, indent=2)
    with open(metric_path, "w") as fd:
        fd.write(results_info)
    print(f"\nResults saved to {metric_path}")
