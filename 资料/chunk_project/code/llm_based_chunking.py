""" 6、模型切分：让 LLM 找出话题切换点，据此切分文本。

思路：
  1. 文本按空行切成段落；
  2. 从当前起点向后累积段落，直到接近预算，形成一个「窗口」；
  3. 把窗口内带编号的段落交给 LLM，问「从哪一段开始内容明显变了」；
  4. 就在那个位置切开，下一块的起点设为它，回到第 2 步。

模型：阿里云百炼 qwen3.8-flash（OpenAI 兼容模式）
文本：模型切分-原始文本.txt

"""

import re
from pathlib import Path
from models import call_qwen

# 提示词
PROMPT = """下面是一组带编号的段落，选自一篇连续的叙事文本，编号是段落在全文中的真实序号。

请找出其中第一处内容相对前文明显发生变化的位置，也就是话题、场景或叙述重点发生切换的那一段。
注意两点：
1. 不要选列表中的第一段，它只是当前窗口的起点；
2. 如果窗口内所有段落都围绕同一话题、没有明显切换，请回答：Answer: none

{body}

只输出一行，格式为：Answer: <段落编号> 或 Answer: none"""


def load_paragraphs(path):
    """读文件，跳过 # 开头的小标题，按空行切段。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    text = "\n".join(l for l in lines if not l.startswith("#"))
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def ask_llm(paragraphs, ids):
    """问模型窗口内哪一段是话题切换点，返回该段下标；"""
    body = "\n\n".join(f"{i}. {paragraphs[i]}" for i in ids)
    nums = re.findall(r"\d+", call_qwen(PROMPT.format(body=body)))
    return int(nums[0]) if nums else None


# 边界切分
def split_by_boundary(paragraphs, budget, find_boundary=ask_llm):
    """滑窗推进：在窗口内判一次边界，从边界处开始下一块。"""
    chunks, start = [], 0
    while start < len(paragraphs):
        end, size = start, 0
        while end < len(paragraphs) and size < budget:
            size += len(paragraphs[end])
            end += 1
        if end - start > 1:       # 窗口内至少两段才值得判
            b = find_boundary(paragraphs, list(range(start, end)))
            end = b if b and start < b <= end else end
        chunks.append((start, end - 1))
        start = end
    return chunks


if __name__ == "__main__":
    paragraphs = load_paragraphs("../docs/模型切分-原始文本.txt")
    total = sum(map(len, paragraphs))
    print(f"{len(paragraphs)} 段，共 {total} 字，窗口 {500} 字\n")

    fp = open('../outputs/llm_based_result.txt', 'a', encoding="utf-8")
    for n, (a, b) in enumerate(split_by_boundary(paragraphs, 500), 1):
        text = "".join(paragraphs[a:b + 1])
        print(f"块 {n}   段 {a}-{b}   {len(text)} 字")
        print(f"        {text[:56]}…\n")

        fp.write(f"块 {n}   段 {a}-{b}   {len(text)} 字")
        fp.write(f"        {text[:56]}…\n")


