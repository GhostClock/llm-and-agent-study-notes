
from typing import List, Dict, Tuple
import numpy as np
import re
import ollama

# 安装
# pip install ollama

'''5、语义分块:
分块策略：
    对⽂本先做句级切分，计算句⼦或短段的向量表示；
    当相邻语义的相似度显著下降（发⽣“语义突变”）时设为切分点。

参数调优说明（仅参考）
    阈值的含义：语义变化敏感度控制器，越低越容易切、越高越保守。
设定方式：
    绝对阈值：例如使用余弦相似度，若 sim < 0.75 则切分（需按语料校准）。
    相对阈值：对全篇的相似度/新颖度分布估计均值μ与标准差σ，使用 μ ± λσ 作为阈值，更稳健。
初始的配置建议（仅限于中文技术/说明文档）：
    窗口大小 window_size：2–4 句
    最小/最大块长：min_chunk_chars=300–400，max_chunk_chars=1000–1200
    阈值策略：novelty > μ + 0.8σ 或相似度 < μ - 0.8σ（先粗调后微调）
    overlap：10% 左右或按“附加上一句”做轻量轮次重叠
'''

def split_sentences_zh(text: str) -> List[str]:
    # 简易中文分句，可替换为 HanLP/Stanza 更稳健的实现
    pattern = re.compile(r'([^。！？；]*[。！？；]+|[^。！？；]+$)')
    return [m.group(0).strip()  for m in pattern.finditer(text)  if m.group(0).strip()]


def rolling_mean(vecs: np.ndarray, i: int, w: int) -> np.ndarray:
    s = max(0, i - w)
    e = min(len(vecs), i + w + 1)
    return vecs[s:e].mean(axis=0)


def semantic_chunk(
        text: str,
        model_name: str = "bge-m3",
        window_size: int = 2,
        min_chars: int = 3,
        max_chars: int = 110,
        lambda_std: float = 0.8,
        overlap_chars: int = 8,
) -> List[Dict]:
    sents = split_sentences_zh(text)
    print("sents:", sents)
    if not sents:
        return []

    # 使用ollama
    emb = ollama.embed(model_name, sents)['embeddings']
    emb = np.asarray(emb)
    print("emb.shape:", emb.shape)

    # 基于窗口均值的“新颖度”分数
    novelties = []
    for i in range(len(sents)):
        ref = rolling_mean(emb, i- 1, window_size) if i > 0 else emb[0]  # 得到窗口
        ref = ref / (np.linalg.norm(ref) + 1e-8)  # 归一化
        novelty = 1.0 - float(np.dot(emb[i], ref))
        novelties.append(novelty)
    novelties = np.array(novelties)
    # print("novelties:", novelties)  # 新颖度

    # 相对阈值：μ + λσ
    mu, sigma = float(novelties.mean()), float(novelties.std() + 1e-8)
    threshold = mu + lambda_std * sigma
    # print("threshold: ", threshold)  相对阈值

    chunks, buf, start_idx = [], "", 0

    def flush(end_idx: int):
        nonlocal buf, start_idx
        if buf.strip():
            chunks.append({
                "text": buf.strip(),
                "meta": {"start_sent": start_idx, "end_sent": end_idx - 1}
            })
        buf, start_idx = "", end_idx

    for i, s in enumerate(sents):
        # 若超长则先冲洗
        if len(buf) + len(s) > max_chars and len(buf) >= min_chars:
            flush(i)
            # 结构化重叠：附加上一个块的尾部
            if overlap_chars > 0 and len(s) < overlap_chars:
                buf = s
                continue

        buf += s

        # 达到最小长度后遇到突变则切分
        if len(buf) >= min_chars and novelties[i] > threshold:
            flush(i + 1)

    if buf:
        flush(len(sents))

    return chunks



if __name__ == '__main__':

    fp = open('../docs/语义切分-原始文本.txt', "r", encoding="utf-8")
    text = fp.read()

    chunks = semantic_chunk(text, window_size=2, min_chars=20, max_chars=1000)
    print('*' * 100)
    for chunk in chunks:
        print(chunk['text'])
        print('=' * 100)



