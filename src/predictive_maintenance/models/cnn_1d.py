"""
1D Convolutional Neural Network (PyTorch) for raw vibration sequence classification.
Processes normalized vibration waveforms directly without manual feature extraction.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from ..logging_config import setup_logger

logger = setup_logger(__name__)


class VibrationSequenceDataset(Dataset):
    """PyTorch Dataset for raw vibration windows."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        # sequences shape: (N, sequence_length) or (N, 1, sequence_length)
        if sequences.ndim == 2:
            sequences = np.expand_dims(sequences, axis=1)
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class Conv1DNet(nn.Module):
    """Compact 1D Convolutional Neural Network for mechanical vibration classification."""

    def __init__(self, num_classes: int = 2, input_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)


class CNN1DClassifier:
    """Wrapper managing training, early stopping, and inference for Conv1DNet."""

    def __init__(
        self,
        num_classes: int = 2,
        input_channels: int = 1,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        max_epochs: int = 30,
        patience: int = 6,
        device: str = "auto",
    ):
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = Conv1DNet(num_classes=num_classes, input_channels=input_channels).to(self.device)
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_f1": []}
        self.metadata: Dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CNN1DClassifier":
        """Trains Conv1DNet with early stopping based on validation loss."""
        if class_names:
            self.classes = class_names
        else:
            self.classes = sorted(list(np.unique(y_train)))

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        y_tr_num = np.array([self.class_to_idx[y] if isinstance(y, str) else y for y in y_train])
        train_dataset = VibrationSequenceDataset(X_train, y_tr_num)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            y_v_num = np.array([self.class_to_idx[y] if isinstance(y, str) else y for y in y_val])
            val_dataset = VibrationSequenceDataset(X_val, y_v_num)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        logger.info(f"Training 1D CNN on {self.device} for up to {self.max_epochs} epochs...")

        for epoch in range(self.max_epochs):
            self.model.train()
            total_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(batch_y)

            avg_train_loss = total_train_loss / len(train_dataset)
            self.history["train_loss"].append(avg_train_loss)

            if val_loader:
                self.model.eval()
                total_val_loss = 0.0
                val_preds = []
                val_targets = []
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        logits = self.model(batch_x)
                        loss = criterion(logits, batch_y)
                        total_val_loss += loss.item() * len(batch_y)
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        val_preds.extend(preds)
                        val_targets.extend(batch_y.cpu().numpy())

                avg_val_loss = total_val_loss / len(val_dataset)
                val_macro_f1 = float(f1_score(val_targets, val_preds, average="macro"))
                self.history["val_loss"].append(avg_val_loss)
                self.history["val_f1"].append(val_macro_f1)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.metadata = metadata or {}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for raw vibration windows."""
        self.model.eval()
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)

        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return np.array([self.idx_to_class[p] for p in preds])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns softmax class probabilities."""
        self.model.eval()
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        if X.ndim == 2:
            X = np.expand_dims(X, axis=1)

        tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def save(self, filepath: Path) -> Path:
        """Saves model weights and class mapping."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "classes": self.classes,
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "history": self.history,
            "metadata": self.metadata,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"CNN1D checkpoint saved to {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: Path, device: str = "cpu") -> "CNN1DClassifier":
        """Loads model weights and metadata from checkpoint."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        instance = cls(
            num_classes=checkpoint["num_classes"],
            input_channels=checkpoint["input_channels"],
            device=device,
        )
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.classes = checkpoint["classes"]
        instance.class_to_idx = checkpoint["class_to_idx"]
        instance.idx_to_class = checkpoint["idx_to_class"]
        instance.history = checkpoint.get("history", {})
        instance.metadata = checkpoint.get("metadata", {})
        instance.model.eval()
        return instance
