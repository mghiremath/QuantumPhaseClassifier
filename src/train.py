from models.mlp import MLPModel
from models.cnn import CNNModel
from utils import get_image_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

def train_model(model_type, epochs=10, batch_size=32, lr=1e-3, device='mps'):

    # CNN model
    if model_type == 'CNN':
        train_loader, test_loader = get_image_dataloaders("../data", batch_size=batch_size, flatten=False)
        model = CNNModel(num_classes=3).to(device)

    # MLP model
    elif model_type == 'MLP':
        train_loader, test_loader = get_image_dataloaders("../data", batch_size=batch_size, flatten=True)
        model = MLPModel(input_dim=370*370, hidden_dim=512, num_classes=3).to(device)    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    test_accuracy = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed.")

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        test_accuracy.append(acc * 100)
        print(f"Test Accuracy: {acc*100:.2f}%")

    epochs_list = range(1, len(test_accuracy) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs_list, test_accuracy, marker='o', linewidth=2)
    plt.xticks(epochs_list)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'{model_type} Test Accuracy with Masked Label Box')
    plt.ylim(85, 101)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{model_type}_accuracy_plot.png")
    plt.show()
    
    
if __name__ == "__main__":
    train_model('MLP')