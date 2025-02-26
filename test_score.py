# coding=utf-8
import json
# import sys
# import re
import numpy as np
# from text2vec import SentenceModel, semantic_search, Similarity
from text2vec import SentenceModel, semantic_search


# simModel_path = './pre_train_model/text2vec-base-chinese'  # 相似度模型
model_name = "shibing624/text2vec-base-chinese"
simModel = SentenceModel(model_name_or_path=model_name, device='cuda:0')


def calc_jaccard(list_a, list_b, threshold=0.3):
    '''
    Jaccard 相似度计算
    '''
    size_b = len(list_b)
    list_c = [i for i in list_a if i in list_b]
    size_c = len(list_c)
    score = size_c / (size_b + 1e-6)
    if score > threshold:
        return 1
    else:
        return 0


def report_score(gold_path, predict_path):
    '''
    计算LLM生成的回答对于标准回答的相似度得分
    '''
    # 读取标准回答
    gold_info = json.load(open(gold_path))
    # 读取LLM生成的回答
    pred_info = json.load(open(predict_path))

    idx = 0
    for gold, pred in zip(gold_info, pred_info):
        question = gold["question"]
        keywords = gold["keywords"]
        gold = gold["answer"].strip()
        pred = pred["answer_4"].strip()  # 只评估合并两路召回且重排序得到的answer
        # 如果标准回答为"无答案"，LLM生成的回答必须也是 "无答案"，否则score为0
        if gold == "无答案" and pred != gold:
            score = 0.0
        elif gold == "无答案" and pred == gold:
            score = 1.0
        else:
            # 计算语义相似度（text2vec 模型）
            semantic_score = semantic_search(simModel.encode([gold]), simModel.encode(pred), top_k=1)[0][0]['score']
            # 计算关键词匹配得分
            join_keywords = [word for word in keywords if word in pred]
            keyword_score = calc_jaccard(join_keywords, keywords)
            # 最终得分 = 语义相似度 50% + 关键词匹配 50%
            score = 0.5 * keyword_score + 0.5 * semantic_score

        # 存储得分和LLM生成的回答
        gold_info[idx]["score"] = score
        gold_info[idx]["predict"] = pred
        idx += 1

        print(f"预测: {question}, 得分: {score}")

    return gold_info


if __name__ == "__main__":
    '''
      Online evaluation
    '''

    # 标准回答路径
    gold_path = "./data/gold.json"
    print("Read gold from %s" % gold_path)

    # 生成回答的路径
    predict_path = "./data/result.json"
    print("Read predict file from %s" % predict_path)

    # 计算对每一个问题的回答得分
    results = report_score(gold_path, predict_path)

    # 计算平均得分
    final_score = np.mean([item["score"] for item in results])
    print("\n")
    print("="*100)
    print(f"预测问题数：{len(results)}, 预测平均得分：{final_score}")
    print("="*100)

    # 存储结果
    metric_path = "./data/metrics.json"
    results_info = json.dumps(results, ensure_ascii=False, indent=2)
    with open(metric_path, "w") as fd:
        fd.write(results_info)
    print(f"\n结果文件保存至{metric_path}")
