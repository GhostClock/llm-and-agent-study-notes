
"""
    1. 定长切分（Fixed-size Chunking）
"""
# 按长度切
def fixed_chunk(text: str, size: int = 40, overlap: int =10):
    """ 按字符定长切分 """
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks


if __name__ == '__main__':
    text = """向量检索的第一步，是把文档切分成一个个块。块的大小直接决定检索的质量上限。如果块切得太大，无关内容会稀释掉关键信息，模型的注意力被摊薄。反过来，如果块切得太小，语义又会被边界拦腰截断，指代关系随之丢失。理想情况下，召回单元应当尽可能接近回答单元，粒度对上了，精确率和召回率才可能同时好看。这就要求切分的时候理解文本自身的语义结构，而不是简单地按固定长度均匀分割。定长切分恰恰是后者，它不看内容，只看长度，所以实现最简单，也最容易切坏。在真实语料上，它产生的块首尾常常是残句。"""
    print(fixed_chunk(text))
