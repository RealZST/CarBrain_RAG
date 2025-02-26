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