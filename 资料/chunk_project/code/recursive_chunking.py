
"""
    3. 递归切分（Recursive Chunking）
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


# 递归切分
def recursive_splitter(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=10,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        keep_separator=False,
        length_function=len,
    )
    return splitter.split_text(text)


if __name__ == '__main__':

    text = '''向量检索的第一步，是把文档切分成一个个块。
    
    块的大小直接决定检索的质量上限。如果块切得太大，无关内容会稀释掉关键信息，模型的注意力被摊薄。反过来，如果块切得太小，语义又会被边界拦腰截断，指代关系随之丢失。理想情况下，召回单元应当尽可能接近回答单元，粒度对上了，精确率和召回率才可能同时好看。
    
    这就要求切分的时候理解文本自身的语义结构，而不是简单地按固定长度均匀分割。'''

    print(recursive_splitter(text))


