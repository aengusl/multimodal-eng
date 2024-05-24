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
checkpoint = "/root/multimodal-eng/models/24may-1"
model = load_checkpoint_and_dispatch(model, checkpoint, device_map=device_map, dtype=TORCH_TYPE)
model = model.eval()
print("Model loaded.")

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

image_folder = "/root/multimodal-eng/pics_of_mates_faces/"
user_input = "Describe this person to me."

images_and_responses = []


for idx, image_file in tqdm(enumerate(os.listdir(image_folder))):
    if not image_file.endswith(".png"):
        continue
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path).convert('RGB')

    history = []

    input_by_model = model.build_conversation_input_ids(
        tokenizer,
        query=user_input,
        images=[image],
        template_version='chat'
    )
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("<|end_of_text|>")[0]
        print("\nCogVLM2:", response)
    images_and_responses.append((
        image_path,
        response
    ))

def create_image_grid(images_and_responses, rows, cols, thumbnail_size=(200, 200), font_size=20):
    grid_width = cols * thumbnail_size[0]
    grid_height = rows * (thumbnail_size[1] + font_size * 4)  # Increase font size multiplier for better text wrapping
    grid_img = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))

    draw = ImageDraw.Draw(grid_img)
    font = ImageFont.load_default()

    for idx, (image_path, response) in enumerate(images_and_responses):
        row, col = divmod(idx, cols)
        x = col * thumbnail_size[0]
        y = row * (thumbnail_size[1] + font_size * 4)  # Increase font size multiplier for better text wrapping
        
        # Open the image again to ensure it's a PIL Image object
        image = Image.open(image_path).convert('RGB')
        image.thumbnail(thumbnail_size)
        grid_img.paste(image, (x, y))

        # Wrap the response text
        text_y = y + thumbnail_size[1] + 5
        wrapped_text = textwrap.fill(response, width=30)  # Adjust width as needed for better wrapping
        draw.text((x, text_y), wrapped_text, fill=(0, 0, 0), font=font)

    return grid_img


# Define the number of rows and columns for the grid
num_images = len(images_and_responses)
rows = (num_images // 3) + (1 if num_images % 3 != 0 else 0)
cols = 3

# Create the image grid
grid_image = create_image_grid(images_and_responses, rows, cols)

# Save the final image
output_path = "/root/multimodal-eng/output/grid_with_responses.jpg"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
grid_image.save(output_path)
print(f"Image grid with responses saved to {output_path}")

