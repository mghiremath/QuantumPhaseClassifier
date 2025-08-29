from models.mlp import MLPModel
from models.cnn import CNNModel
from models.vit import ViTModel
from utils import get_image_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

def train_model(model_type, epochs=10, batch_size=32, lr=1e-3, device='mps', task='classification'):

    # CNN model
    if model_type == 'CNN':
        train_loader, test_loader = get_image_dataloaders(
            "../data", batch_size=batch_size, flatten=False,
            target_type='class' if task == 'classification' else 'regression'
        )
        out_dim = 3 if task == 'classification' else 1
        model = CNNModel(out_dim=out_dim).to(device)

    # MLP model
    elif model_type == 'MLP':
        train_loader, test_loader = get_image_dataloaders(
            "../data", batch_size=batch_size, flatten=True,
            target_type='class' if task == 'classification' else 'regression'
        )
        out_dim = 3 if task == 'classification' else 1
        model = MLPModel(input_dim=370*370, hidden_dim=512, out_dim=out_dim).to(device)
    
    # ViT model
    elif model_type == 'ViT':
        train_loader, test_loader = get_image_dataloaders(
            "../data", batch_size=batch_size, flatten=False,
            target_type='class' if task == 'classification' else 'regression'
        )
        out_dim = 3 if task == 'classification' else 1
        model = ViTModel(
            img_size=370, patch_size=37, in_chans=1, out_dim=out_dim,
            embed_dim=64, depth=3, num_heads=2,
            mlp_ratio=3, dropout=0.2
        ).to(device)
        # stronger regularization for ViT
        weight_decay = 1e-4
    else:
        raise ValueError("model_type must be one of: 'CNN', 'MLP', 'ViT'")
    
    if task == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    test_metric_history = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            if task == 'classification':
                loss = criterion(out, yb)
            else:
                # out: [B, 1] -> squeeze to [B]
                loss = criterion(out.squeeze(1), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed.")

        # Evaluate
        model.eval()
        with torch.no_grad():
            if task == 'classification':
                correct, total = 0, 0
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
                acc = correct / total
                test_metric_history.append(acc * 100)
                print(f"Test Accuracy: {acc*100:.2f}%")
            else:
                # compute MAE on test set
                total_abs_err, total = 0.0, 0
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb).squeeze(1)
                    total_abs_err += (preds - yb).abs().sum().item()
                    total += yb.size(0)
                mae = total_abs_err / total
                test_metric_history.append(mae)
                print(f"Test MAE: {mae:.4f}")

    epochs_list = range(1, len(test_metric_history) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs_list, test_metric_history, marker='o', linewidth=2)
    plt.xticks(epochs_list)
    plt.xlabel('Epoch')
    if task == 'classification':
        plt.ylabel('Test Accuracy (%)')
        plt.title(f'{model_type} Test Accuracy with Masked Label Box')
        plt.ylim(85, 101)
    else:
        plt.ylabel('Test MAE')
        plt.title(f'{model_type} Test MAE with Masked Label Box')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    if task == 'classification':
        plt.savefig(f"results/{model_type}_accuracy_plot.png")
    else:
        plt.savefig(f"results/{model_type}_mae_plot.png")
    plt.show()
    
    
if __name__ == "__main__":
    train_model('ViT', task='regression')
