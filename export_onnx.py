import torch
from bcresnet import BCResNets

# These must match exactly how you trained (from your training config)
TAU = 1.0
NUM_CLASSES = 3          # silence, unknown, wakeword
USE_SSN = False           # you trained with --no_ssn
DROPOUT = 0.3
USE_ARCFACE = False
N_MELS = 40
SPEC_WIDTH = 101          # from your training log: Spec Shape [1, 1, 40, 101]

device = torch.device("cpu")

# Build the same empty model structure
model = BCResNets(tau=TAU, num_classes=NUM_CLASSES, use_ssn=USE_SSN,
                   dropout=DROPOUT, use_arcface=USE_ARCFACE).to(device)

# Load your trained weights into it
checkpoint = torch.load("best_bcresnet.pth", map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint  # it's already a plain state dict

model.load_state_dict(state_dict)
model.eval()

# A fake input just to show ONNX the model's input shape
dummy_input = torch.randn(1, 1, N_MELS, SPEC_WIDTH, requires_grad=False)

torch.onnx.export(
    model,
    dummy_input,
    "bcresnet_astra.onnx",
    input_names=["mel_spectrogram"],
    output_names=["logits"],
    opset_version=13,
)

print("Saved bcresnet_astra.onnx")