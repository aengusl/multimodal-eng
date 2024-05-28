"""
This is a demo for using CogVLM2 in CLI using multi-GPU with lower memory.
If your single GPU is not enough to drive this model, you can use this demo to run this model on multiple graphics cards with limited video memory.
Here, we default that your graphics card has 24GB of video memory, which is not enough to load the FP16 / BF16 model.
so , need to use two graphics cards to load. We set '23GiB' for each GPU to avoid out of memory.
"""
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import textwrap
from jsonformer.format import highlight_values
from jsonformer.main import Jsonformer

from transformers import AutoModelForCausalLM, AutoTokenizer

from subclass_jsonformer import CogVLMJsonformer

MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

device_map = infer_auto_device_map(
    model=model,
    max_memory={i: "23GiB" for i in range(torch.cuda.device_count())},
    # set 23GiB for each GPU, depends on your GPU memory, you can adjust this value
    no_split_module_classes=["CogVLMDecoderLayer"]
)
checkpoint = "/root/multimodal-eng/models/28may-1"
model = load_checkpoint_and_dispatch(model, checkpoint, device_map=device_map, dtype=TORCH_TYPE)
model = model.eval()
print("Model loaded.")


car = {
  "type": "object",
  "properties": {
    "car": {
      "type": "object",
      "properties": {
        "make": {"type": "string"},
        "model": {"type": "string"},
        "year": {"type": "number"},
        "colors": {
          "type": "array",
          "items": {"type": "string"}
        },
        "features": {
          "type": "object",
          "properties": {
            "audio": {
              "type": "object",
              "properties": {
                "brand": {"type": "string"},
                "speakers": {"type": "number"},
                "hasBluetooth": {"type": "boolean"}
              }
            },
            "safety": {
              "type": "object",
              "properties": {
                "airbags": {"type": "number"},
                "parkingSensors": {"type": "boolean"},
                "laneAssist": {"type": "boolean"}
              }
            },
            "performance": {
              "type": "object",
              "properties": {
                "engine": {"type": "string"},
                "horsepower": {"type": "number"},
                "topSpeed": {"type": "number"}
              }
            }
          }
        }
      }
    },
    "owner": {
      "type": "object",
      "properties": {
        "firstName": {"type": "string"},
        "lastName": {"type": "string"},
        "age": {"type": "number"},
      }
    }
  }
}

# builder = Jsonformer(
builder = CogVLMJsonformer(
    model=model,
    tokenizer=tokenizer,
    json_schema=car,
    prompt="Generate an example car",
)

print("Generating...")
output = builder()

highlight_values(output)
