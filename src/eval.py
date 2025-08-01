from models.mlp import MLPModel
from utils import get_image_dataloaders
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_mlp(checkpoint_path, batch_size=32, device='cpu'):
    _, test_loader = get_image_dataloaders("../data", batch_size=batch_size, flatten=True)
    model = MLPModel(input_dim=370*370, hidden_dim=512, num_classes=3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    acc = sum([p==t for p,t in zip(all_preds, all_labels)]) / len(all_preds)
    print(f"Test Accuracy: {acc*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate_mlp("mlp_checkpoint.pt")