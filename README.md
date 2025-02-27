# CarBrain-RAG
## **Environment setup**

查看cuda版本

```
module avail 2>&1 | grep -i cuda
```

把cuda和gcc load进来：

```
module load cuda/12.2.1
module load gcc
```

安装依赖

```
pip install -r requirements.txt
```

## Get started

运行run.sh

```
sbatch run.sh
```

结果保存在./data/result.json中

evaluate结果：

```
sbatch test_score.sh
```


## Project Structure

```
├── Dockerfile                     # 镜像文件
├── README.md                      # 说明文档
├── bm25_retriever.py              # BM25召回
├── build.sh                       # 镜像编译打包
├── data                           # 数据目录
│   ├── result.json                # 结果提交文件
│   ├── test_question.json         # 测试集
│   └── train_a.pdf                # 训练集
├── faiss_retriever.py             # faiss向量召回
├── vllm_model.py                  # vllm大模型加速wrapper
├── pdf_parse.py                   # pdf文档解析器
├── pre_train_model                # 预训练大模型
│   ├── Qwen-7B-Chat               # Qwen-7B
│   │   └── download.py
│   ├── bge-reranker-large         # bge重排序模型
│   └── m3e-large                  # 向量检索模型
├── qwen_generation_utils.py       # qwen答案生成的工具函数
├── requirements.txt               # 此项目的第三方依赖库
├── rerank_model.py                # 重排序逻辑
├── run.py                         # 主文件
└── run.sh                         # 主运行脚本
```