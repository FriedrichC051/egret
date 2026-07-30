"""最小连通性测试：确认 OPENAI_API_KEY 读到、openai 库在、Kimi 接口能返回。"""
import os
import sys

# Kimi 是国内域名，必须绕开系统代理(Clash)，否则连接失败
for _p in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_p, None)

key = os.environ.get("OPENAI_API_KEY")
print("OPENAI_API_KEY:", "SET(len=%d)" % len(key) if key else "UNSET")
if not key:
    sys.exit("未读到 key")

try:
    from openai import OpenAI
except Exception as e:
    sys.exit("openai 库不可用: %r" % e)

base_url = "https://api.moonshot.cn/v1"
model = "kimi-k2.7-code"
try:
    client = OpenAI(base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "只回复两个字：可用"}],
        temperature=1,
        max_tokens=10,
    )
    print("MODEL:", model)
    print("REPLY:", resp.choices[0].message.content)
    print("API_OK")
except Exception as e:
    print("API_ERROR:", repr(e))
    sys.exit(1)
