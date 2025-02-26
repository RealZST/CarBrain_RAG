"""
 基于 vLLM 加载 Qwen-7B-Chat（通义千问 7B）模型，用于 批量推理，并支持 停止词过滤、采样策略、显存管理等功能。
"""

import os
import torch
import time

from config import *
from vllm import LLM, SamplingParams

# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
from transformers import GenerationConfig
# from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids
from qwen_generation_utils import make_context, get_stop_words_ids


# 关闭 transformers 分词器的多线程，减少多线程冲突问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

# Qwen模型的停止标记
IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"


# 释放gpu上没有用到的显存以及显存碎片
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            # 释放未被占用但仍然缓存的 GPU 显存
            torch.cuda.empty_cache()
            # 清理 CUDA IPC（进程间通信）对象，减少显存碎片
            torch.cuda.ipc_collect()


class ChatLLM(object):
    '''
    封装 Qwen-7B-Chat，实现 LLM 推理
    自动管理 tokenizer、停止词、显存等
    支持批量推理
    '''
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            pad_token='<|extra_0|>',
            eos_token='<|endoftext|>',
            padding_side='left',
            trust_remote_code=True
        )
        # 加载 Qwen-7B-Chat 的生成配置
        self.generation_config = GenerationConfig.from_pretrained(model_path, pad_token_id=self.tokenizer.pad_token_id)
        self.tokenizer.eos_token_id = self.generation_config.eos_token_id
        self.stop_words_ids = []

        # 加载vLLM
        self.model = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=1,  # 如果是多卡，可以自己把这个并行度设置为卡数N
            trust_remote_code=True,
            gpu_memory_utilization=0.6,  # 可以根据gpu的利用率自己调整这个比例
            dtype="bfloat16"
        )

        # 获取 停止词 ID
        for stop_id in get_stop_words_ids(self.generation_config.chat_format, self.tokenizer):
            self.stop_words_ids.extend(stop_id)
        self.stop_words_ids.extend([self.generation_config.eos_token_id])

        # LLM的采样参数
        sampling_kwargs = {
            "stop_token_ids": self.stop_words_ids,
            "early_stopping": False,
            "top_p": 1.0,
            "top_k": -1 if self.generation_config.top_k == 0 else self.generation_config.top_k,
            "temperature": 0.0,
            "max_tokens": 2000,
            "repetition_penalty": self.generation_config.repetition_penalty,
            "n": 1,  # 每次生成 n 个不同的答案并全部返回
            "best_of": 2,  # 生成 best_of 个答案 只返回最好的一个
            "use_beam_search": True
        }
        self.sampling_params = SamplingParams(**sampling_kwargs)

    # 批量推理，输入一个batch，返回一个batch的答案
    def infer(self, prompts):
        batch_text = []
        for q in prompts:
            # make_context() 格式化 Qwen 的输入
            raw_text, _ = make_context(
                self.tokenizer,
                q,
                system="You are a helpful assistant.",
                max_window_size=self.generation_config.max_window_size,
                chat_format=self.generation_config.chat_format,
            )
            batch_text.append(raw_text)

        # vLLM 批量推理
        outputs = self.model.generate(batch_text, sampling_params=self.sampling_params)

        # 去除停止标记
        batch_response = []
        for output in outputs:
            output_str = output.outputs[0].text
            if IMEND in output_str:
                output_str = output_str[:-len(IMEND)]
            if ENDOFTEXT in output_str:
                output_str = output_str[:-len(ENDOFTEXT)]
            batch_response.append(output_str)

        # 释放显存
        torch_gc()

        return batch_response


if __name__ == "__main__":
    # qwen7 = "./pre_train_model/Qwen-7B-Chat/qwen/Qwen-7B-Chat"
    qwen7 = "Qwen/Qwen-7B-Chat"
    start = time.time()
    llm = ChatLLM(qwen7)
    test = ["吉利汽车座椅按摩", "吉利汽车语音组手唤醒", "自动驾驶功能介绍"]
    generated_text = llm.infer(test)
    print(generated_text)
    end = time.time()
    print("cost time: " + str((end-start)/60))
