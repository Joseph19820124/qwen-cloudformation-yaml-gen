import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指向你刚才下载模型的文件夹路径，通常在当前目录下的 Qwen/Qwen3-14B-AWQ
model_path = "../Qwen/Qwen3-14B-AWQ"

print("🚀 正在将 Qwen3-14B 注入 A10G 显存...")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型：注意针对 G5 的优化参数
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, # G5 必须用 bfloat16 以获得最佳性能
    device_map="auto",          # 自动分配显存
    trust_remote_code=True
)

# 准备测试问题：结合你的 Nike 和吉隆坡背景
prompt = "你现在是 Nike 的首席市场官。请分析：如果我们要针对吉隆坡的网球爱好者推广一款高性能球鞋，Qwen3-14B 这种 AI 能力能如何在我们的电商转化中发挥作用？"

messages = [
    {"role": "system", "content": "You are a senior executive at Nike specializing in AI and E-commerce strategy."},
    {"role": "user", "content": prompt}
]

# 转换格式
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

print("\n--- 🏁 Qwen3-14B 开始推理 (观察你的仪表盘) ---\n")

# 开始生成
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# 解码并输出
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response.split("assistant")[-1].strip())
