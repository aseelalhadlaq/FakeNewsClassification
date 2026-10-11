# ============================================================
# 8. DA-MiniLM++ MLP classifier and MiniLM logistic baseline
# ============================================================

class EmbeddingDataset(Dataset):
    def __init__(self, X, y=None, sample_weight=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
        self.sample_weight = None if sample_weight is None else torch.tensor(sample_weight, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {"x": self.X[idx]}
        if self.y is not None:
            item["y"] = self.y[idx]
        if self.sample_weight is not None:
            item["sample_weight"] = self.sample_weight[idx]
        return item


class DAMiniLMMLP(nn.Module):
    def __init__(self, input_dim, num_classes=NUM_CLASSES, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def predict_mlp(model, X, batch_size=512):
    model.eval()
    ds = EmbeddingDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(DEVICE)
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.vstack(probs)


def train_da_minilm_mlp(X_train, y_train, w_train, X_valid, y_valid, X_test, seed=42):
    set_all_seeds(seed)
    input_dim = X_train.shape[1]
    model = DAMiniLMMLP(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(reduction="none")

    train_ds = EmbeddingDataset(X_train, y_train, w_train)
    valid_ds = EmbeddingDataset(X_valid, y_valid)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)

    best_state = None
    best_valid_f1 = -1.0
    patience = 6
    bad_epochs = 0
    max_epochs = 35

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            x = batch["x"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            sw = batch["sample_weight"].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss_vec = criterion(logits, y)
            loss = (loss_vec * sw).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(x)

        valid_proba = predict_mlp(model, X_valid)
        valid_pred = valid_proba.argmax(axis=1)
        valid_f1 = f1_score(y_valid, valid_pred, average="macro", zero_division=0)

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    valid_proba = predict_mlp(model, X_valid)
    test_proba = predict_mlp(model, X_test)
    return valid_proba, test_proba, float(best_valid_f1)


def train_logreg_baseline(X_train, y_train, w_train, X_valid, y_valid, X_test, seed=42):
    clf = LogisticRegression(
        max_iter=4000,
        C=2.0,
        solver="lbfgs",
        multi_class="auto",
        random_state=seed,
    )
    clf.fit(X_train, y_train, sample_weight=w_train)
    valid_proba = clf.predict_proba(X_valid)
    test_proba = clf.predict_proba(X_test)
    return valid_proba, test_proba