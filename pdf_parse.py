#!/usr/bin/env python
# coding: utf-8

"""
解析 PDF 文件，并按规则对文本进行分块。
实现了3种分块方法：
- ParseBlock()：按页解析 PDF 内容，并按照一定规则分块（1）遇到一些具体字样/符号；（2）文本字体大小变化
- ParseOnePageWithRule()：按页解析 PDF 内容，按句号和max_seq划分文本
- ParseAllPage()：将整个 PDF 看作一个整体，然后使用滑动窗口进行分块
"""

import pdfplumber
from PyPDF2 import PdfReader


class DataProcess(object):

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.data = []

    #  数据过滤，拆分长文本并过滤无用字符，然后将拆分后的文本存入self.data
    def Datafilter(self, line, header, pageid, max_seq=1024):
        sz = len(line)
        if sz < 6:  # 去掉短句（长度 < 6 的句子被丢弃）
            return

        if sz > max_seq:  # 拆分长文本，按以下符号进行分割
            if "■" in line:
                sentences = line.split("■")
            elif "•" in line:
                sentences = line.split("•")
            elif "\t" in line:
                sentences = line.split("\t")
            else:
                sentences = line.split("。")

            for subsentence in sentences:
                # subsentence = subsentence.replace("\n", "")
                if 5 <= len(subsentence) <= max_seq:
                    # 过滤无用字符
                    subsentence = subsentence.replace(",", "").replace("\n", "").replace("\t", "")
                    # 存入 self.data
                    if subsentence not in self.data:
                        self.data.append(subsentence)
        else:
            # 过滤无用字符
            line = line.replace("\n", "").replace(",", "").replace("\t", "")
            # 存入 self.data
            if (line not in self.data):
                self.data.append(line)

    # 提取一级标题或页头
    def GetHeader(self, page):
        try:
            lines = page.extract_words()[::]
        except:
            return None
        if len(lines) > 0:
            for line in lines:  # 每个line是一个单词的字典（dict），包含该单词的位置信息
                if ("目录" in line["text"] or ".........." in line["text"]):
                    # 排除目录页
                    return None
                # 一级标题
                if 17 < line["top"] < 20:
                    return line["text"]
            # 如果没有找到符合条件的标题，返回页面的第一个单词
            return lines[0]["text"]
        # 空白页
        return None

    # 按页解析 PDF 内容，按以下规则分块：（1）遇到一些具体字样/符号；（2）文本字体大小变化
    def ParseBlock(self, max_seq=1024):
        with pdfplumber.open(self.pdf_path) as pdf:
            # 逐页遍历 PDF 内容
            for i, p in enumerate(pdf.pages):
                # 返回一级标题或页面的第一个单词
                header = self.GetHeader(p)
                # 排除目录页和空白页
                if header is None:
                    continue

                # ’p‘这页不是目录页或空白页
                # 按照文本流方式组合文本，保持自然阅读顺序
                # 提取文本的字体大小信息，用于后续段落识别
                texts = p.extract_words(use_text_flow=True, extra_attrs=["size"])[::]

                squence = ""  # 用于存储当前段落的文本内容
                lastsize = 0  # 记录上一个text的字体大小，方便判断是否换段
                for idx, line in enumerate(texts):
                    # 跳过第1段text
                    if idx < 1:
                        continue
                    # 如果第2段是纯数字，也跳过
                    if idx == 1:
                        if line["text"].isdigit():
                            continue
                    
                    cursize = line["size"]  # 当前text的字体大小
                    text = line["text"]
                    # 跳过特殊符号
                    if (text == "□" or text == "•"):
                        continue
                    # 遇到下面这些字样时，认为是新的段落
                    elif (text == "警告！" or text == "注意！" or text == "说明！"):
                        if len(squence) > 0:
                            # 将当前 squence 传给 Datafilter() 进行存储
                            self.Datafilter(squence, header, i, max_seq=max_seq)
                        # 清空 squence 以开始新段落
                        squence = ""
                    # 若当前文本的字体大小没有变化，则属于同一段落，因此 拼接 squence
                    elif (format(lastsize, ".5f") == format(cursize, ".5f")):
                        if len(squence) > 0:
                            squence = squence + text
                        else:
                            squence = text
                    else:
                        lastsize = cursize
                        # 如果 squence 长度较短（小于 15），则继续拼接
                        if 0 < len(squence) < 15:
                            squence = squence + text
                        else:
                            # 如果 squence 长度大于 15，则传给 Datafilter() 进行存储
                            if len(squence) > 0:
                                self.Datafilter(squence, header, i, max_seq=max_seq)
                            squence = text
                # 处理最后的squence
                if len(squence) > 0:
                    self.Datafilter(squence, header, i, max_seq=max_seq)

    # 按页解析 PDF 内容，按句号和max_seq划分文档
    def ParseOnePageWithRule(self, max_seq=512, min_len=6):
        # 逐页解析 PDF
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                # 去掉两端空格、\t、\n
                text = word.strip()
                if ("...................." in text or "目录" in text):
                    continue
                if len(text) < 1:
                    continue
                if text.isdigit():
                    continue
                # 按照上面的规则过滤完后，将这页的文本再拼回到一起，page_content代表整页文本
                page_content = page_content + text
            
            # 若整页文本过短，则跳过
            if len(page_content) < min_len:
                continue
            # 如果整页文本长度小于 max_seq，则直接存入 self.data。
            if len(page_content) < max_seq:
                if page_content not in self.data:
                    self.data.append(page_content)
            # 如果整页文本长度大于等于 max_seq，则按句号切分
            else:
                sentences = page_content.split("。")
                cur = ""
                for idx, sentence in enumerate(sentences):
                    if len(cur + sentence) >= max_seq:
                        if (cur + sentence) not in self.data:
                            self.data.append(cur + sentence)
                        # cur = sentence
                        cur = ""
                    else:
                        cur = cur + sentence

    # 滑动窗口功能实现
    def SlidingWindow(self, sentences, kernel=512, stride=1):
        # sz = len(sentences)
        cur = ""
        fast = 0  # 滑动窗口的右边界，遍历 sentences
        slow = 0  # 滑动窗口的左边界，控制窗口的滑动起点
        while (fast < len(sentences)):
            sentence = sentences[fast]
            if len(cur + sentence) >= kernel:
                if (cur + sentence) not in self.data:
                    self.data.append(cur + sentence + "。")
                cur = cur[len(sentences[slow] + "。"):]
                slow = slow + 1
            cur = cur + sentence + "。"
            fast = fast + 1

    #  滑窗法提取段落
    #  1. 把pdf看做一个整体,作为一个字符串
    #  2. 利用句号当做分隔符,切分成一个数组
    #  3. 利用滑窗法对数组进行滑动, 当窗口内文本的长度大于max_seq时进行分块
    def ParseAllPage(self, max_seq=512, min_len=6):
        all_content = ""
        # 逐页解析 PDF
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                # 去掉两端空格、\t、\n
                text = word.strip()
                if ("...................." in text or "目录" in text):
                    continue
                if len(text) < 1:
                    continue
                if text.isdigit():
                    continue
                # 按照上面的规则过滤完后，将这页的文本再拼回到一起，page_content代表整页文本
                page_content = page_content + text
            
            # 若整页文本过短，则跳过
            if len(page_content) < min_len:
                continue
            
            # 拼接每页文本，all_content代表整个pdf文件的所有文本
            all_content = all_content + page_content
        
        # 按句号切分文本
        sentences = all_content.split("。")
        
        self.SlidingWindow(sentences, kernel=max_seq)


if __name__ == "__main__":
    dp = DataProcess(pdf_path="./data/train_a.pdf")
    
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)
    print(len(dp.data))
    
    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)
    print(len(dp.data))
    
    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)
    print(len(dp.data))
    
    data = dp.data
    out = open("all_text.txt", "w")
    for line in data:
        line = line.strip("\n")  # 去除 line 两端的换行符 \n
        out.write(line)
        out.write("\n")
    out.close()
