
import os
from openai import OpenAI


def call_qwen(prompt=""):
    """调用 Qwen；传入 user_prompt 时使用 system/user 双消息。"""
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 需要在环境变量增加evn字段，如果本地调试也可以把SK写死到这里
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    messages = (
        [
            {"role": "user", "content": prompt},
        ]
    )
    response = client.chat.completions.create(
        model="qwen3.8-max",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content or ""


