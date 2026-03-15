
import sys
import torch
from torchvision import transforms
from PIL import Image

sys.path.append('./src')
from model import build_model

CLASS_NAMES = ['Healthy', 'Diseased']

predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])


def predict(image_path, model_path='./outputs/best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(num_classes=2, device=device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    image  = Image.open(image_path).convert("RGB")
    tensor = predict_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    probs      = torch.softmax(output, dim=1)[0]
    pred_idx   = probs.argmax().item()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item() * 100

    return pred_class, confidence


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/leaf_image.jpg")
    else:
        image_path = sys.argv[1]
        pred, conf = predict(image_path)
        print(f"Prediction : {pred}")
        print(f"Confidence : {conf:.2f}%")
