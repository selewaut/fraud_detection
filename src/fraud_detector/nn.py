import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from fraud_detector.utils import load_data
from fraud_detector.utils import data_preprocessing

# Load data
df = load_data()

df = data_preprocessing(df)

# Features and target
features = [
    "edge_noise",
    "text_density",
    "grayscale_variance",
    "alpha_channel_density",
    "unique_font_colors",
    "reported_income",
]

df = df.dropna(subset="label")  # Drop rows with NaN in features or label
X = df[features]
y = df["label"]

# Encode label: CLEAN=0, EDITED=1

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Now y is numpy array of 0/1

# Impute missing values (mean)
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train/test split (stratified for imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Class distribution in train: {np.bincount(y_train)}")


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # For BCE loss

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Create datasets
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

# DataLoaders (batch size small for small dataset)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class SimpleNN(nn.Module):
    def __init__(self, input_size=6):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 150),
            nn.ReLU(),
            nn.Dropout(0.2),  # Dropout to prevent overfitting on small data
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1),  # logits output, in inference mode will apply sigmoid.
        )

    def forward(self, x):
        return self.fc(x)


# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)
criterion = nn.BCEWithLogitsLoss()  # For binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)


def train_model(
    model, train_loader, test_loader, criterion, optimizer, epochs=200, patience=10
):
    best_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validate mode
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()

        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))
    return train_losses, val_losses


def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    y_proba = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)  # Probabilities
            preds = (probs > 0.5).float()  # Binary predictions

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(probs.cpu().numpy())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_proba = np.array(y_proba).flatten()

    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)

    print(f"Test Set Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"PR AUC: {pr_auc:.3f}")
    print(f"F1 Score: {f1:.3f}")


# Train
train_losses, val_losses = train_model(
    model, train_loader, test_loader, criterion, optimizer
)

evaluate_model(model, train_loader)
evaluate_model(model, test_loader)

import matplotlib.pyplot as plt


def plot_decision_boundary(ax, X, y, model, title, columns):
    ax.scatter(X[columns[0]], X[columns[1]], c=y, cmap="coolwarm", edgecolors="k", s=50)

    # Create grid (use original scaled data ranges)
    x_min, x_max = X[columns[0]].min() - 1, X[columns[0]].max() + 1
    y_min, y_max = X[columns[1]].min() - 1, X[columns[1]].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Prepare full grid with mean values for other columns (already scaled)
    grid = np.c_[xx.ravel(), yy.ravel()]
    full_grid = np.full((grid.shape[0], len(features)), 0.0)  # Placeholder
    col_idx0 = features.index(columns[0])
    col_idx1 = features.index(columns[1])
    full_grid[:, col_idx0] = grid[:, 0]
    full_grid[:, col_idx1] = grid[:, 1]
    for i, col in enumerate(features):
        if i not in [col_idx0, col_idx1]:
            full_grid[:, i] = X_train[:, i].mean()  # Use train mean (scaled)

    # Predict with PyTorch
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.tensor(full_grid, dtype=torch.float32).to(device)
        Z = torch.sigmoid(model(grid_tensor)).cpu().numpy() > 0.5
        Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    plt.colorbar(
        ax.collections[0], ax=ax, label="Label"
    )  # Add colorbar to each subplot


# Usage with subplots
X_train_df = pd.DataFrame(X_train, columns=features)
X_test_df = pd.DataFrame(X_test, columns=features)

from itertools import combinations

cols = ["alpha_channel_density", "edge_noise", "grayscale_variance"]


for pair in combinations(cols, 2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns

    plot_decision_boundary(
        ax1, X_train_df, y_train, model, "Decision Boundary on Training Set", pair
    )
    plot_decision_boundary(
        ax2, X_test_df, y_test, model, "Decision Boundary on Test Set", pair
    )

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
