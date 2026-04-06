import re

import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "qwen/Qwen3-4B"


def clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^```(?:yaml|yml)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    yaml_starts = [
        "AWSTemplateFormatVersion:",
        "Description:",
        "Parameters:",
        "Resources:",
    ]
    start_positions = [text.find(x) for x in yaml_starts if x in text]
    if start_positions:
        text = text[min(start_positions):]

    stop_markers = [
        "\nExplanation:",
        "\nHere is",
        "\nThis template",
        "\nNote:",
        "\nThe template",
    ]
    stop_positions = [text.find(x) for x in stop_markers if x in text]
    if stop_positions:
        text = text[:min(stop_positions)]

    return text.strip()


def main() -> None:
    model_dir = snapshot_download(MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AWS Cloud Architect. "
                "Output only valid AWS CloudFormation YAML. "
                "Do not explain anything. "
                "Do not include markdown fences. "
                "Do not include any text before or after the YAML."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write a complete AWS CloudFormation YAML template.\n"
                "Requirements:\n"
                "1. A DynamoDB table named NikeInventory.\n"
                "2. An IAM Role for a Lambda function.\n"
                "3. A Lambda function that can read and write to the table.\n"
                "4. An API Gateway REST API as the trigger.\n"
                "5. Include Lambda invoke permission for API Gateway.\n"
                "6. Use inline Lambda code in ZipFile.\n"
                "Return YAML only."
            ),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("\n--- Generating CloudFormation template ---")

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1400,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    response_ids = generated_ids[:, model_inputs.input_ids.shape[1] :]
    response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0].strip()
    print(clean_output(response))


if __name__ == "__main__":
    main()
