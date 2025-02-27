#!/usr/bin/env python
# coding: utf-8

"""
Parse PDF files and segment text based on predefined rules.
Implements three segmentation methods:
- ParseBlock(): Parses PDF content page by page and segments text based on: 
(1) specific symbols;
(2) changes in font size.
- ParseOnePageWithRule(): Parses PDF content page by page and segments text based on sentence-ending periods and max_seq length.
- ParseAllPage(): Treats the entire PDF as a single text unit and segments it using a sliding window.
"""

import pdfplumber
from PyPDF2 import PdfReader


class DataProcess(object):

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.data = []

    # Data filtering: Splits long text, removes unnecessary characters,
    # and stores the processed text in self.data
    def Datafilter(self, line, header, pageid, max_seq=1024):
        sz = len(line)
        if sz < 6:  # Discard short sentences (length < 6)
            return

        if sz > max_seq:  # Split long text using the following symbols
            if "■" in line:
                sentences = line.split("■")
            elif "•" in line:
                sentences = line.split("•")
            elif "\t" in line:
                sentences = line.split("\t")
            else:
                sentences = line.split("。")

            for subsentence in sentences:
                if 5 <= len(subsentence) <= max_seq:
                    # Remove unnecessary characters
                    subsentence = subsentence.replace(",", "").replace("\n", "").replace("\t", "")
                    # Store in self.data
                    if subsentence not in self.data:
                        self.data.append(subsentence)
        else:
            # Remove unnecessary characters
            line = line.replace("\n", "").replace(",", "").replace("\t", "")
            # Store in self.data
            if (line not in self.data):
                self.data.append(line)

    # 提取一级标题或页头
    def GetHeader(self, page):
        try:
            lines = page.extract_words()[::]
        except:
            return None
        if len(lines) > 0:
            for line in lines:  # Each line is a dictionary
                if ("目录" in line["text"] or ".........." in line["text"]):
                    # Exclude table of contents pages
                    return None
                # Primary title
                if 17 < line["top"] < 20:
                    return line["text"]
            # If no title is found, return the first word on the page
            return lines[0]["text"]
        # Blank page
        return None

    # Parse PDF content page by page and segment text based on:
    # (1) specific symbols; (2) changes in font size.
    def ParseBlock(self, max_seq=1024):
        with pdfplumber.open(self.pdf_path) as pdf:
            # Iterate through each page in the PDF
            for i, p in enumerate(pdf.pages):
                # Get primary title or first word on the page
                header = self.GetHeader(p)
                # Exclude table of contents and blank pages
                if header is None:
                    continue

                # 'p' is neither a table of contents page nor a blank page
                # Extract text in a natural reading order
                # Extract font size information for paragraph identification
                texts = p.extract_words(use_text_flow=True, extra_attrs=["size"])[::]

                squence = ""  # Stores the current paragraph content
                lastsize = 0  # Stores the font size of the previous text for paragraph identification
                for idx, line in enumerate(texts):
                    # Skip the first segment
                    if idx < 1:
                        continue
                    # Skip if the second segment is purely digital
                    if idx == 1:
                        if line["text"].isdigit():
                            continue

                    cursize = line["size"]  # Current text font size
                    text = line["text"]
                    # Skip special symbols
                    if (text == "□" or text == "•"):
                        continue
                    # If encountering these keywords, start a new paragraph
                    elif (text == "警告！" or text == "注意！" or text == "说明！"):
                        if len(squence) > 0:
                            # Store the current sequence
                            self.Datafilter(squence, header, i, max_seq=max_seq)
                        # Start a new paragraph
                        squence = ""
                    # If the font size remains the same, append to the same paragraph
                    elif (format(lastsize, ".5f") == format(cursize, ".5f")):
                        if len(squence) > 0:
                            squence = squence + text
                        else:
                            squence = text
                    else:
                        lastsize = cursize
                        # If squence length is short (<15), continue appending
                        if 0 < len(squence) < 15:
                            squence = squence + text
                        else:
                            # If squence length is >15, store it
                            if len(squence) > 0:
                                self.Datafilter(squence, header, i, max_seq=max_seq)
                            squence = text
                # Process the last squence
                if len(squence) > 0:
                    self.Datafilter(squence, header, i, max_seq=max_seq)

    # Parse PDF content page by page, segment text based on periods and max_seq
    def ParseOnePageWithRule(self, max_seq=512, min_len=6):
        # Iterate through each page in the PDF
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                # Remove leading and trailing spaces, tabs, and newline characters
                text = word.strip()
                if ("...................." in text or "目录" in text):
                    continue
                if len(text) < 1:
                    continue
                if text.isdigit():
                    continue
                # After filtering based on the above rules, concatenate the text of the page.
                # `page_content` represents the full text of the current page.
                page_content = page_content + text

            # Skip the page if the total text is too short
            if len(page_content) < min_len:
                continue
            # If the total page text length is less than max_seq, store it directly in self.data
            if len(page_content) < max_seq:
                if page_content not in self.data:
                    self.data.append(page_content)
            # If the total page text length is greater than or equal to max_seq, split it by periods
            else:
                sentences = page_content.split("。")
                cur = ""
                for idx, sentence in enumerate(sentences):
                    if len(cur + sentence) >= max_seq:
                        if (cur + sentence) not in self.data:
                            self.data.append(cur + sentence)
                        cur = ""
                    else:
                        cur = cur + sentence

    def SlidingWindow(self, sentences, kernel=512, stride=1):
        cur = ""
        fast = 0  # Right boundary of the sliding window, iterating through sentences
        slow = 0  # Left boundary controlling window movement
        while (fast < len(sentences)):
            sentence = sentences[fast]
            if len(cur + sentence) >= kernel:
                if (cur + sentence) not in self.data:
                    self.data.append(cur + sentence + "。")
                cur = cur[len(sentences[slow] + "。"):]
                slow = slow + 1
            cur = cur + sentence + "。"
            fast = fast + 1

    # Sliding window-based segmentation
    # 1. Treat the entire PDF as a single text string
    # 2. Use periods as delimiters to split the text into an array
    # 3. Slide the window over the array and segment the text when its length exceeds max_seq
    def ParseAllPage(self, max_seq=512, min_len=6):
        all_content = ""
        # 逐页解析 PDF
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                text = word.strip()
                if ("...................." in text or "目录" in text):
                    continue
                if len(text) < 1:
                    continue
                if text.isdigit():
                    continue
                # After filtering based on the above rules, concatenate the text of the page.
                # `page_content` represents the full text of the current page.
                page_content = page_content + text

            # Skip the page if the total text is too short
            if len(page_content) < min_len:
                continue

            # Concatenate the text from each page
            # 'all_content' represents the entire text of the PDF file
            all_content = all_content + page_content

        # Use periods as delimiters to split the text
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
        line = line.strip("\n")
        out.write(line)
        out.write("\n")
    out.close()
