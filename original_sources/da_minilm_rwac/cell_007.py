"""
da_minilm_rwc_model.py

Class 2 of 3: the model(s) only -- frozen MiniLM sentence embeddings, the
DA-MiniLM-r MLP classifier, and the frozen-embedding Logistic Regression
comparator (MiniLM-LogReg-RW). No splitting logic (see temporal_splitter
.py) and no experiment orchestration -- sample weighting, calibration,
multi-seed batteries, and bootstrap comparison (see
da_minilm_experiment_runner.py).

This class trains and predicts. It does not decide what data to train on,
how to weight it, or how to calibrate the output -- those are the
experiment runner's job, so this class stays reusable for other weighting
or calibration schemes without modification.
"""

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


NUM_CLASSES_DEFAULT = 6


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EmbeddingDataset(Dataset):
    def __init__(self, X, y=None, sample_weight=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
        if sample_weight is None:
            sample_weight = np.ones(len(X), dtype=np.float32)
        self.sample_weight = torch.tensor(sample_weight, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        item = {"x": self.X[idx], "sample_weight": self.sample_weight[idx]}
        if self.y is not None:
            item["y"] = self.y[idx]
        return item


class DAMiniLMClassifier(nn.Module):
    """The MLP head trained on frozen MiniLM embeddings."""

    def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES_DEFAULT, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def predict_da_minilm_proba(model, loader, device) -> np.ndarray:
    model.eval()
    probas = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            logits = model(x)
            probas.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probas, axis=0)


class DAMiniLMModel:

    def __init__(
        self,
        output_dir="./da_minilm_results",
        minilm_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        num_classes: int = NUM_CLASSES_DEFAULT,
        hidden_dim: int = 256,
        dropout: float = 0.25,
        mlp_epochs: int = 35,
        mlp_batch_size: int = 128,
        mlp_lr: float = 1e-3,
        mlp_weight_decay: float = 1e-4,
        mlp_patience: int = 6,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.minilm_model_name = minilm_model_name
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.mlp_epochs = mlp_epochs
        self.mlp_batch_size = mlp_batch_size
        self.mlp_lr = mlp_lr
        self.mlp_weight_decay = mlp_weight_decay
        self.mlp_patience = mlp_patience

        self._embedding_cache = {}

    # ---------------- Embedding extraction ----------------

    def encode_texts_minilm(self, texts: Sequence[str], cache_key: str, batch_size: int = 128) -> np.ndarray:
        """Encode texts once, cached in memory and on disk. Requires
        downloading `self.minilm_model_name` from the Hugging Face Hub the
        first time it's called for a given environment."""
        cache_path = self.output_dir / f"embeddings_{cache_key}.npy"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        if cache_path.exists():
            X = np.load(cache_path)
            self._embedding_cache[cache_key] = X
            print(f"Loaded cached embeddings: {cache_path} {X.shape}")
            return X

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.minilm_model_name, device=str(self.device))
        X = model.encode(
            list(texts), batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)

        np.save(cache_path, X)
        self._embedding_cache[cache_key] = X
        print(f"Saved embeddings: {cache_path} {X.shape}")
        return X

    # ---------------- DA-MiniLM MLP ----------------

    def train_da_minilm_mlp(self, X_train, y_train, w_train, X_valid, y_valid, X_test, seed: int = 42):
        """Trains the MLP head on frozen embeddings with per-sample
        weights (`w_train`), early-stopping on validation macro-F1.
        Returns (valid_proba, test_proba, best_valid_macro_f1) -- raw,
        uncalibrated probabilities; calibration is the experiment
        runner's job, not this class's."""
        set_all_seeds(seed)

        train_ds = EmbeddingDataset(X_train, y_train, w_train)
        valid_ds = EmbeddingDataset(X_valid, y_valid)
        test_ds = EmbeddingDataset(X_test, None)

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(train_ds, batch_size=self.mlp_batch_size, shuffle=True, generator=generator)
        valid_loader = DataLoader(valid_ds, batch_size=self.mlp_batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.mlp_batch_size, shuffle=False)

        model = DAMiniLMClassifier(
            input_dim=X_train.shape[1], num_classes=self.num_classes,
            hidden_dim=self.hidden_dim, dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_weight_decay)

        best_state = None
        best_valid_macro_f1 = -1.0
        bad_epochs = 0

        for epoch in range(self.mlp_epochs):
            model.train()
            for batch in train_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                sw = batch["sample_weight"].to(self.device)

                logits = model(x)
                per_sample_loss = F.cross_entropy(logits, y, reduction="none")
                loss = (per_sample_loss * sw).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            valid_proba = predict_da_minilm_proba(model, valid_loader, self.device)
            valid_pred = valid_proba.argmax(axis=1)
            valid_macro_f1 = f1_score(y_valid, valid_pred, average="macro", zero_division=0)

            if valid_macro_f1 > best_valid_macro_f1:
                best_valid_macro_f1 = valid_macro_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= self.mlp_patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        valid_proba = predict_da_minilm_proba(model, valid_loader, self.device)
        test_proba = predict_da_minilm_proba(model, test_loader, self.device)
        return valid_proba, test_proba, best_valid_macro_f1

    # ---------------- Frozen-embedding Logistic Regression comparator ----------------

    def train_logreg_baseline(self, X_train, y_train, w_train, X_valid, y_valid, X_test, seed: int = 42):
        """MiniLM-LogReg-RW: frozen MiniLM embeddings + Logistic
        Regression. class_weight=None because sample_weight (w_train)
        already encodes class balance if the caller included it (see
        da_minilm_experiment_runner.py's combined_sample_weights)."""
        clf = make_pipeline(
            StandardScaler(with_mean=True),
            LogisticRegression(max_iter=2000, class_weight=None, random_state=seed),
        )
        clf.fit(X_train, y_train, logisticregression__sample_weight=w_train)
        return clf.predict_proba(X_valid), clf.predict_proba(X_test)


if __name__ == "__main__":
    print("da_minilm_model.py loaded. Class: DAMiniLMModel")