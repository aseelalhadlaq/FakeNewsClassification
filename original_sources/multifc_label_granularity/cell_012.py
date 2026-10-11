# ============================================================
# 1d. DAMiniLMModel  (dataset-agnostic; identical logic to LIAR2 main model)
# ============================================================
class _EmbeddingDataset(Dataset):
    def __init__(self, X, y=None, sample_weight=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
        if sample_weight is None:
            sample_weight = np.ones(len(X), dtype=np.float32)
        self.sample_weight = torch.tensor(sample_weight, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        item = {"x": self.X[i], "sample_weight": self.sample_weight[i]}
        if self.y is not None:
            item["y"] = self.y[i]
        return item


class _DAMiniLMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.25):
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


class DAMiniLMModel:
    """Frozen-MiniLM encoder + DA-MiniLM MLP head + LogReg estimator + weighting +
    two-stage calibration + metrics. One instance per class-granularity."""

    def __init__(self, output_dir, num_classes,
                 minilm_model_name=MINILM_MODEL_NAME,
                 device=DEVICE):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = int(num_classes)
        self.minilm_model_name = minilm_model_name
        self.device = device
        self._embedding_cache = {}          # shared across instances by the runner
        self._sentence_model = None
        # calibration / weighting grids
        self.class_weight_beta = CLASS_WEIGHT_BETA
        self.tau_grid = TAU_GRID
        self.temperature_grid = TEMPERATURE_GRID

    # ---------- encoding ----------
    def encode_texts_minilm(self, texts, cache_key, batch_size=EMBED_BATCH_SIZE):
        cache_path = self.output_dir / f"embeddings_{cache_key}.npy"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        if cache_path.exists():
            X = np.load(cache_path)
            self._embedding_cache[cache_key] = X
            print(f"Loaded cached embeddings: {cache_path} {X.shape}")
            return X
        if self._sentence_model is None:
            self._sentence_model = SentenceTransformer(self.minilm_model_name,
                                                       device=str(self.device))
        X = self._sentence_model.encode(
            list(texts), batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        np.save(cache_path, X)
        self._embedding_cache[cache_key] = X
        print(f"Saved embeddings: {cache_path} {X.shape}")
        return X

    # ---------- weighting ----------
    @staticmethod
    def _normalize_weights(w, min_w=0.05, max_w=20.0):
        w = np.clip(np.asarray(w, float), min_w, max_w)
        return w / (w.mean() + 1e-12)

    def _effective_number_class_weights(self, y):
        counts = np.bincount(np.asarray(y, int),
                             minlength=self.num_classes).astype(float)
        w = np.ones(self.num_classes, float)
        for c in range(self.num_classes):
            n = counts[c]
            if n > 0:
                eff = (1.0 - self.class_weight_beta ** n) / (1.0 - self.class_weight_beta)
                w[c] = 1.0 / eff
        return w / (w.mean() + 1e-12)

    def class_sample_weights(self, y_train):
        cw = self._effective_number_class_weights(y_train)
        return self._normalize_weights(cw[np.asarray(y_train, int)])

    def recency_sample_weights(self, time_rank_train, lam=0.0):
        t = np.asarray(time_rank_train, float)
        age = t.max() - t
        return self._normalize_weights(np.exp(-float(lam) * age * 10.0))

    def combined_sample_weights(self, y_train, time_rank_train, recency_lambda=0.0):
        return self._normalize_weights(
            self.class_sample_weights(y_train)
            * self.recency_sample_weights(time_rank_train, recency_lambda))

    # ---------- calibration primitives ----------
    @staticmethod
    def _softmax(logits):
        logits = np.asarray(logits, float)
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / (e.sum(axis=1, keepdims=True) + 1e-12)

    @staticmethod
    def _proba_to_logits(p, eps=1e-12):
        return np.log(np.clip(p, eps, 1.0))

    def logit_adjustment_bias(self, y_train, tau=1.0, eps=1e-12):
        counts = np.bincount(np.asarray(y_train, int),
                             minlength=self.num_classes).astype(float)
        prior = counts / counts.sum()
        return tau * np.log(prior + eps)

    def apply_bias_and_temperature(self, proba, bias=None, temperature=1.0):
        logits = self._proba_to_logits(proba)
        if bias is not None:
            logits = logits - bias.reshape(1, -1)
        logits = logits / max(float(temperature), 1e-6)
        return self._softmax(logits)

    def select_best_calibration(self, valid_proba, y_valid, y_train,
                                objective="macro_f1", calibration_mode="two_stage"):
        """Two-stage selection: choose tau (logit adjustment) and temperature on the
        validation set. Stage 1 fixes tau by macro-F1; stage 2 fixes temperature given
        that tau. This avoids the temperature non-identifiability that arises when the
        full (tau, temperature) grid is scored jointly under a macro-F1 objective, where
        temperature is a monotone rescaling that leaves argmax — and hence macro-F1 —
        unchanged, so the joint search locks temperature to the first grid value by an
        iteration-order artifact rather than by fit."""
        y_valid = np.asarray(y_valid); y_train = np.asarray(y_train)

        def _score(cal):
            pred = cal.argmax(axis=1)
            if objective == "macro_f1":
                return f1_score(y_valid, pred, average="macro", zero_division=0)
            if objective == "nll":
                return -self.safe_nll(y_valid, cal)
            raise ValueError(objective)

        if calibration_mode == "two_stage":
            # ---- stage 1: pick tau at temperature = 1.0 by the label-aware objective
            best_tau, best_tau_score = 0.0, -1e9
            for tau in self.tau_grid:
                bias = self.logit_adjustment_bias(y_train, tau=tau)
                cal = self.apply_bias_and_temperature(valid_proba, bias=bias,
                                                      temperature=1.0)
                s = _score(cal)
                if s > best_tau_score:
                    best_tau_score, best_tau = s, tau
            # ---- stage 2: pick temperature under NLL (a proper scoring rule that IS
            #      sensitive to temperature), holding tau fixed
            bias = self.logit_adjustment_bias(y_train, tau=best_tau)
            best_temp, best_temp_score = 1.0, -1e9
            for temp in self.temperature_grid:
                cal = self.apply_bias_and_temperature(valid_proba, bias=bias,
                                                      temperature=temp)
                s = -self.safe_nll(y_valid, cal)   # stage-2 objective is NLL by design
                if s > best_temp_score:
                    best_temp_score, best_temp = s, temp
            return {"tau": best_tau, "temperature": best_temp,
                    "score": float(best_tau_score)}

        # ---- legacy single-stage joint grid (kept for comparison / ablation) ----
        best = {"tau": 0.0, "temperature": 1.0, "score": -1e9}
        for tau in self.tau_grid:
            bias = self.logit_adjustment_bias(y_train, tau=tau)
            for temp in self.temperature_grid:
                cal = self.apply_bias_and_temperature(valid_proba, bias=bias,
                                                      temperature=temp)
                s = _score(cal)
                if s > best["score"]:
                    best = {"tau": tau, "temperature": temp, "score": float(s)}
        return best

    # ---------- estimators ----------
    def train_da_minilm_mlp(self, X_tr, y_tr, w_tr, X_va, y_va, X_te, seed=42):
        set_all_seeds(seed)
        tr_ds = _EmbeddingDataset(X_tr, y_tr, w_tr)
        va_ds = _EmbeddingDataset(X_va, y_va)
        te_ds = _EmbeddingDataset(X_te, None)
        g = torch.Generator(); g.manual_seed(seed)
        tr_ld = DataLoader(tr_ds, batch_size=MLP_BATCH_SIZE, shuffle=True, generator=g)
        va_ld = DataLoader(va_ds, batch_size=MLP_BATCH_SIZE, shuffle=False)
        te_ld = DataLoader(te_ds, batch_size=MLP_BATCH_SIZE, shuffle=False)

        model = _DAMiniLMClassifier(input_dim=X_tr.shape[1],
                                    num_classes=self.num_classes).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=MLP_LR,
                                weight_decay=MLP_WEIGHT_DECAY)
        best_state, best_f1, bad = None, -1.0, 0
        for _ in range(MLP_EPOCHS):
            model.train()
            for b in tr_ld:
                x = b["x"].to(self.device); y = b["y"].to(self.device)
                sw = b["sample_weight"].to(self.device)
                loss = (F.cross_entropy(model(x), y, reduction="none") * sw).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            va_p = self._predict_proba(model, va_ld)
            f1 = f1_score(y_va, va_p.argmax(1), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, bad = f1, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= MLP_PATIENCE:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return (self._predict_proba(model, va_ld),
                self._predict_proba(model, te_ld), best_f1)

    def _predict_proba(self, model, loader):
        model.eval(); out = []
        with torch.no_grad():
            for b in loader:
                out.append(F.softmax(model(b["x"].to(self.device)), dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def train_logreg(self, X_tr, y_tr, w_tr, X_va, X_te, seed=42):
        clf = make_pipeline(
            StandardScaler(with_mean=True),
            # NOTE: the `multi_class` argument was removed in scikit-learn >=1.7;
            # multinomial is the default for softmax-capable solvers, so it is omitted
            # for forward/backward compatibility (behaviour is unchanged).
            LogisticRegression(max_iter=2000, class_weight=None,
                               random_state=seed, n_jobs=-1),
        )
        clf.fit(X_tr, y_tr, logisticregression__sample_weight=w_tr)
        return clf.predict_proba(X_va), clf.predict_proba(X_te)

    # ---------- metrics ----------
    @staticmethod
    def ordinal_mae(y, p):
        return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))

    @staticmethod
    def within_one_accuracy(y, p):
        return float(np.mean(np.abs(np.asarray(y) - np.asarray(p)) <= 1))

    @staticmethod
    def expected_calibration_error(y, proba, n_bins=10):
        y = np.asarray(y); proba = np.asarray(proba)
        conf = proba.max(1); pred = proba.argmax(1)
        correct = (pred == y).astype(float)
        ece = 0.0; edges = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            if m.sum() == 0:
                continue
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        return float(ece)

    def multiclass_brier(self, y, proba):
        y = np.asarray(y)
        oh = np.eye(self.num_classes)[y]
        return float(np.mean(np.sum((proba - oh) ** 2, axis=1)))

    def safe_nll(self, y, proba):
        return float(log_loss(y, np.clip(proba, 1e-12, 1.0),
                              labels=list(range(self.num_classes))))

    def evaluate(self, y_true, y_pred, proba=None, **meta):
        row = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(y_true, y_pred, average="macro",
                                               zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro",
                                         zero_division=0),
            "ordinal_mae": self.ordinal_mae(y_true, y_pred),
            "within_one_accuracy": self.within_one_accuracy(y_true, y_pred),
        }
        if proba is not None:
            row["ece"] = self.expected_calibration_error(y_true, proba)
            row["brier"] = self.multiclass_brier(y_true, proba)
            row["nll"] = self.safe_nll(y_true, proba)
        row.update(meta)
        return row
