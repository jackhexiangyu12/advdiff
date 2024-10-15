import torch
from PIL import Image
import requests
from transformers import AutoProcessor, Blip2VisionModelWithProjection

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")
model = Blip2VisionModelWithProjection.from_pretrained(
    "Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16
)
model.to(device)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

outputs = model(**inputs)
image_embeds = outputs.image_embeds
print(image_embeds.shape)