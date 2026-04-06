# qwen-cloudformation-yaml-gen

Minimal script for generating AWS CloudFormation YAML with Qwen3 while forcing non-thinking mode and stripping `<think>` blocks, code fences, and extra explanation text from the output.

## Files

- `generate_cfn_yaml.py`: main generation script
- `requirements.txt`: Python dependencies

## Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_cfn_yaml.py
```

## Notes

- The script defaults to `Qwen/Qwen3-4B`.
- It calls `tokenizer.apply_chat_template(..., enable_thinking=False)` to disable reasoning output.
- It uses inline cleanup logic to keep only the YAML section.
