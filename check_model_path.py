import argparse

from transformers import LlamaForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
args = parser.parse_args()

device_map = "auto"
model = LlamaForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=True,
    device_map=device_map,
)
print(model.base_model_prefix)
