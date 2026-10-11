"""
shuffle_control.py

Shuffle-control (label permutation) validation, addressing this project's
tracked "remaining required experiment": establish a null distribution
for test macro-F1 by training on randomly permuted TRAIN labels (keeping
TEST labels real), repeated 200-500 times, and compare the real model's
test macro-F1 against that null distribution. If the real result isn't
comfortably above the null distribution's upper tail, the model isn't
demonstrably learning anything from the true label structure.

Deliberately model-agnostic: takes a `train_predict_fn` callable rather
than hard-coding DAMiniLMModel, so the same validator works for the
frozen-embedding LogReg baseline (fast, recommended for 200-500 shuffles),
the MLP (much slower -- consider fewer shuffles or a subsample), or any
future model. `train_predict_fn` must accept
(X_train, y_train, X_valid, y_valid, X_test, seed) and return
(valid_proba, test_proba).

Usage
-----
    from shuffle_control import ShuffleControlValidator
    from da_minilm_model import DAMiniLMModel

    model = DAMiniLMModel(output_dir="./da_minilm_results")

    def train_predict_fn(X_train, y_train, X_valid, y_valid, X_test, seed):
        w_train = np.ones(len(y_train))  # shuffle control doesn't reweight
        return model.train_logreg_baseline(X_train, y_train, w_train, X_valid, y_valid, X_test, seed=seed)

    validator = ShuffleControlValidator(train_predict_fn, n_shuffles=300, seed=42)
    result = validator.run(X_train, y_train, X_valid, y_valid, X_test, y_test)
    print(result["summary"])
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


class ShuffleControlValidator:

    def __init__(self, train_predict_fn: Callable, n_shuffles: int = 300, seed: int = 42):
        """
        train_predict_fn(X_train, y_train, X_valid, y_valid, X_test, seed)
            -> (valid_proba, test_proba)
        Any function with this signature works -- e.g. a lambda wrapping
        DAMiniLMModel.train_logreg_baseline() or .train_da_minilm_mlp()
        with fixed/uniform sample weights, since shuffle control tests
        whether the LABEL structure matters, not whether reweighting does.
        """
        self.train_predict_fn = train_predict_fn
        self.n_shuffles = n_shuffles
        self.seed = seed

    def run(
        self, X_train, y_train, X_valid, y_valid, X_test, y_test,
        real_test_macro_f1: Optional[float] = None,
    ) -> dict:
        """Trains `n_shuffles` times on a randomly permuted copy of
        y_train (test labels are never touched), collecting test macro-F1
        each time to build the null distribution. If `real_test_macro_f1`
        isn't supplied, it's computed here by running train_predict_fn
        once more on the REAL (unshuffled) labels, so the comparison is
        always apples-to-apples on the same X_test/y_test.
        """
        rng = np.random.RandomState(self.seed)

        if real_test_macro_f1 is None:
            _, real_test_proba = self.train_predict_fn(X_train, y_train, X_valid, y_valid, X_test, self.seed)
            real_test_macro_f1 = f1_score(y_test, real_test_proba.argmax(axis=1), average="macro", zero_division=0)

        null_scores = np.empty(self.n_shuffles)
        for i in range(self.n_shuffles):
            shuffled_y_train = rng.permutation(y_train)
            shuffle_seed = int(rng.randint(0, 2**31 - 1))
            _, test_proba = self.train_predict_fn(
                X_train, shuffled_y_train, X_valid, y_valid, X_test, shuffle_seed,
            )
            null_scores[i] = f1_score(y_test, test_proba.argmax(axis=1), average="macro", zero_division=0)
            if (i + 1) % max(1, self.n_shuffles // 10) == 0:
                print(f"  shuffle {i + 1}/{self.n_shuffles}: macro_f1={null_scores[i]:.4f}")

        p_value = float((np.sum(null_scores >= real_test_macro_f1) + 1) / (self.n_shuffles + 1))

        summary = {
            "n_shuffles": self.n_shuffles,
            "real_test_macro_f1": float(real_test_macro_f1),
            "null_mean": float(null_scores.mean()),
            "null_std": float(null_scores.std()),
            "null_p95": float(np.percentile(null_scores, 95)),
            "null_p99": float(np.percentile(null_scores, 99)),
            "null_max": float(null_scores.max()),
            "p_value": p_value,
            "real_exceeds_null_p95": bool(real_test_macro_f1 > np.percentile(null_scores, 95)),
            "real_exceeds_null_max": bool(real_test_macro_f1 > null_scores.max()),
        }

        return {"summary": summary, "null_scores": null_scores}

    @staticmethod
    def print_summary(result: dict) -> None:
        s = result["summary"]
        print(f"Real test macro-F1:        {s['real_test_macro_f1']:.4f}")
        print(f"Null distribution (n={s['n_shuffles']}): "
              f"mean={s['null_mean']:.4f}, std={s['null_std']:.4f}, "
              f"95th pct={s['null_p95']:.4f}, 99th pct={s['null_p99']:.4f}, max={s['null_max']:.4f}")
        print(f"p-value (fraction of shuffles >= real): {s['p_value']:.4f}")
        print(f"Real result exceeds null 95th percentile: {s['real_exceeds_null_p95']}")
        print(f"Real result exceeds null maximum:          {s['real_exceeds_null_max']}")


if __name__ == "__main__":
    print("shuffle_control.py loaded. Class: ShuffleControlValidator")