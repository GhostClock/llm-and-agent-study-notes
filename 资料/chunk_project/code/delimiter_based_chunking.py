

"""
    2. 分隔符切分（Delimiter-based Chunking）
"""

# 按分隔符切分
def delimiter_chunk(text: str, sep: str = "\n\n", max_len: int = 1200,
                    keep_sep: bool = True) -> list[str]:
    """按分隔符切分。
            1.只保证切点落在分隔符处，不保证块长。
            2.超长片段可以继续交给递归切分兜底，这里没有处理，如果>max_len直接抛出异常。
    """
    parts = []
    for seg in text.split(sep):
        if not seg.strip():
            continue
        if len(seg) > max_len:
            raise ValueError(f"存在超长片段 {len(seg)} > {max_len}")
        parts.append(seg + (sep if keep_sep else ""))
    return parts


if __name__ == '__main__':
    text = """向量检索的第一步，是把文档切分成一个个块。块的大小直接决定检索的质量上限。如果块切得太大，无关内容会稀释掉关键信息，模型的注意力被摊薄。反过来，如果块切得太小，语义又会被边界拦腰截断，指代关系随之丢失。理想情况下，召回单元应当尽可能接近回答单元。粒度对上了，精确率和召回率才可能同时好看。这就要求切分的时候理解文本自身的语义结构。而不是简单地按固定长度均匀分割。定长切分恰恰是后者，它不看内容，只看长度，所以实现最简单，也最容易切坏。"""

    print(delimiter_chunk(text, sep="。"))