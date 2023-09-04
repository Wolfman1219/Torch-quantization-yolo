import torch
from torch.quantization import quantize_dynamic

# Load the YOLOv8n PyTorch model
model = torch.load('yolov8n.pt')['model'] 

# Configure the model for dynamic quantization
model.eval()
model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Test quantized model
input = torch.randn(1, 3, 640, 640)
# out = model(input)

# Save quantized model
torch.save(model, 'yolov8n_quantized.pt')
