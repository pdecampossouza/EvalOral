"""EvalOral1 evolving neuro-fuzzy core.

Research-oriented extension of the user's EvolvingNeuroFuzzyAdvanced model.
Main additions:
- feature relevance fallback uses 1.0 (fully relevant), not 1/d;
- probability/evidence outputs;
- explicit rule event log and snapshots;
- consequent-compatible rule merging with Jensen-Shannon gate;
- pooled-variance merge for Gaussian premises;
- operational novelty / uncertainty / rule-conflict scores;
- human-readable rule summaries.

This module is experimental research code, not a clinical diagnostic device.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any
import math
import numpy as np


def T_min(z: np.ndarray) -> float:
    return 0.0 if z.size == 0 else float(np.min(z))


def S_max(z: np.ndarray) -> float:
    return 0.0 if z.size == 0 else float(np.max(z))


def C_mean(z: np.ndarray) -> float:
    return 0.0 if z.size == 0 else float(np.mean(z))


@dataclass
class UnimNary:
    e: float = 0.5
    T: Callable[[np.ndarray], float] = T_min
    S: Callable[[np.ndarray], float] = S_max
    C: Callable[[np.ndarray], float] = C_mean

    def aggregate(self, z: np.ndarray) -> Tuple[float, str, float]:
        z = np.asarray(z, dtype=float).ravel()
        if z.size == 0:
            return float(self.e), "COMP", 0.0
        all_le = bool(np.all(z <= self.e))
        all_ge = bool(np.all(z >= self.e))
        if all_le and not all_ge:
            val, regime, rho = self.T(z), "AND", -1.0
        elif all_ge and not all_le:
            val, regime, rho = self.S(z), "OR", +1.0
        else:
            val, regime, rho = self.C(z), "COMP", 0.0
        return float(np.clip(val, 0.0, 1.0)), regime, rho


def gaussian_memberships_per_feature(x: np.ndarray, center: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    center = np.asarray(center, dtype=float).reshape(-1)
    sigma = np.maximum(np.asarray(sigma, dtype=float).reshape(-1), 1e-6)
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in [0, 1] for base-2 logarithms."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    return float(np.clip(_entropy(m) - 0.5 * _entropy(p) - 0.5 * _entropy(q), 0.0, 1.0))


@dataclass
class NeuroFuzzyRule:
    rule_id: int
    center: np.ndarray
    sigma: np.ndarray
    n_features: int
    n_classes: int
    initial_class: int
    e: float = 0.5
    support: int = 1
    birth_step: int = 0
    last_update_step: int = 0
    last_regime: Optional[str] = None
    last_rho: Optional[float] = None
    regime_counts: Dict[str, int] = field(default_factory=lambda: {"AND": 0, "OR": 0, "COMP": 0})

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(-1)
        self.sigma = np.maximum(np.asarray(self.sigma, dtype=float).reshape(-1), 1e-3)
        self.n_features = int(self.n_features)
        self.n_classes = int(self.n_classes)
        self.class_counts = np.zeros(self.n_classes, dtype=float)
        self.class_counts[int(self.initial_class)] += 1.0
        self.unim = UnimNary(e=float(self.e))

    @property
    def class_probs(self) -> np.ndarray:
        total = float(self.class_counts.sum())
        if total <= 0:
            return np.ones(self.n_classes, dtype=float) / self.n_classes
        return self.class_counts / total

    @property
    def dominant_class(self) -> int:
        return int(np.argmax(self.class_probs))

    def activation_details(self, x: np.ndarray, feature_weights: Optional[np.ndarray] = None, record_regime: bool = False) -> Dict[str, Any]:
        x = np.asarray(x, dtype=float).reshape(-1)
        a = gaussian_memberships_per_feature(x, self.center, self.sigma)
        if feature_weights is None:
            w = np.ones(self.n_features, dtype=float)
        else:
            w = np.asarray(feature_weights, dtype=float).reshape(-1)
            if w.size != self.n_features:
                raise ValueError("feature_weights has wrong dimension")
            w = np.clip(w, 0.0, 1.0)
        h = np.clip(w * a + (1.0 - w) * self.unim.e, 0.0, 1.0)
        val, regime, rho = self.unim.aggregate(h)
        self.last_regime, self.last_rho = regime, rho
        if record_regime:
            self.regime_counts[regime] = self.regime_counts.get(regime, 0) + 1
        return {"activation": float(val), "regime": regime, "rho": float(rho), "membership": a, "mixed": h}

    def activation(self, x: np.ndarray, feature_weights: Optional[np.ndarray] = None) -> float:
        return float(self.activation_details(x, feature_weights)["activation"])

    def update_premise(self, x: np.ndarray, step: int, lr_center: float = 0.1) -> None:
        x = np.asarray(x, dtype=float).reshape(-1)
        self.support += 1
        lr = float(max(1.0 / self.support, lr_center))
        old_center = self.center.copy()
        self.center = (1.0 - lr) * self.center + lr * x
        # Smooth an approximate standard deviation; preserve a floor.
        dev = np.abs(x - old_center) + 1e-3
        self.sigma = np.maximum(0.9 * self.sigma + 0.1 * dev, 1e-3)
        self.last_update_step = int(step)

    def update_consequent(self, y: int) -> None:
        self.class_counts[int(y)] += 1.0


class FeatureSeparabilityBuffer:
    """Sliding-window between/within-class feature relevance in [0,1]."""

    def __init__(self, n_features: int, n_classes: int, buffer_size: int = 200, weight_floor: float = 0.0):
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.buffer_size = int(buffer_size)
        self.weight_floor = float(weight_floor)
        self.X_buffer: List[np.ndarray] = []
        self.y_buffer: List[int] = []

    def add_sample(self, x: np.ndarray, y: int) -> None:
        self.X_buffer.append(np.asarray(x, dtype=float).reshape(-1))
        self.y_buffer.append(int(y))
        if len(self.X_buffer) > self.buffer_size:
            self.X_buffer.pop(0)
            self.y_buffer.pop(0)

    def compute_feature_weights(self) -> np.ndarray:
        # IMPORTANT: w is a gate/relevance coefficient, not a simplex weight.
        # Equal/full relevance is therefore w_j = 1, not 1/d.
        if len(self.X_buffer) < 2 or len(set(self.y_buffer)) < 2:
            return np.ones(self.n_features, dtype=float)

        X = np.stack(self.X_buffer, axis=0)
        y = np.asarray(self.y_buffer, dtype=int)
        sep = np.zeros(self.n_features, dtype=float)
        mean_all = X.mean(axis=0)
        for j in range(self.n_features):
            within, between = 0.0, 0.0
            for c in range(self.n_classes):
                xc = X[y == c, j]
                if xc.size == 0:
                    continue
                var_c = float(xc.var(ddof=1)) if xc.size > 1 else 0.0
                within += xc.size * var_c
                between += xc.size * (float(xc.mean()) - float(mean_all[j])) ** 2
            sep[j] = between / within if within > 1e-12 else 0.0
        mx = float(sep.max())
        if mx <= 0.0:
            return np.ones(self.n_features, dtype=float)
        w = sep / mx
        if self.weight_floor > 0:
            w = self.weight_floor + (1.0 - self.weight_floor) * w
        return np.clip(w, 0.0, 1.0)


class EvolvingNeuroFuzzyEvalOral:
    """Auditable evolving classifier for EvalOral1 streams."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        alpha_add: float = 0.4,
        tau_merge: float = 0.9,
        buffer_size_similarity: int = 200,
        buffer_size_separability: int = 200,
        max_rules: Optional[int] = None,
        merge_require_same_class: bool = True,
        merge_max_js: float = 0.15,
        feature_weight_floor: float = 0.05,
        sigma_init: float = 0.5,
        activation_mode: str = "unim",
        label_conflict_policy: str = "adapt",
        label_conflict_min_confidence: float = 0.60,
        random_state: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.alpha_add = float(alpha_add)
        self.tau_merge = float(tau_merge)
        self.buffer_size_similarity = int(buffer_size_similarity)
        self.max_rules = max_rules
        self.merge_require_same_class = bool(merge_require_same_class)
        self.merge_max_js = float(merge_max_js)
        self.sigma_init = float(sigma_init)
        self.activation_mode = str(activation_mode).strip().lower()
        if self.activation_mode not in {"unim", "coverage_gated_unim"}:
            raise ValueError("activation_mode must be 'unim' or 'coverage_gated_unim'")
        self.label_conflict_policy = str(label_conflict_policy).strip().lower()
        if self.label_conflict_policy not in {"adapt", "new_rule"}:
            raise ValueError("label_conflict_policy must be 'adapt' or 'new_rule'")
        self.label_conflict_min_confidence = float(label_conflict_min_confidence)
        self.rng = np.random.default_rng(random_state)
        self.feature_names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(n_features)]
        self.class_names = list(class_names) if class_names is not None else [str(i) for i in range(n_classes)]
        self.rules: List[NeuroFuzzyRule] = []
        self.sep_buffer = FeatureSeparabilityBuffer(n_features, n_classes, buffer_size_separability, feature_weight_floor)
        self.sim_X_buffer: List[np.ndarray] = []
        self.event_log: List[Dict[str, Any]] = []
        self.step = 0
        self._next_rule_id = 1

    def _event(self, kind: str, **kwargs: Any) -> None:
        row = {"step": int(self.step), "event": kind, **kwargs}
        self.event_log.append(row)

    @property
    def feature_weights_(self) -> np.ndarray:
        return self.sep_buffer.compute_feature_weights()

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "EvolvingNeuroFuzzyEvalOral":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y = np.asarray(y, dtype=int).ravel()
        if X.shape[0] != y.size:
            raise ValueError("X and y size mismatch")
        for xi, yi in zip(X, y):
            self._update_one(xi, int(yi))
        return self

    fit = partial_fit

    def _add_similarity_sample(self, x: np.ndarray) -> None:
        self.sim_X_buffer.append(np.asarray(x, dtype=float).reshape(-1))
        if len(self.sim_X_buffer) > self.buffer_size_similarity:
            self.sim_X_buffer.pop(0)

    def _create_rule(self, x: np.ndarray, y: int, reason: str, best_activation: Optional[float] = None) -> None:
        rid = self._next_rule_id
        self._next_rule_id += 1
        rule = NeuroFuzzyRule(
            rule_id=rid,
            center=np.asarray(x, dtype=float).copy(),
            sigma=np.full(self.n_features, self.sigma_init, dtype=float),
            n_features=self.n_features,
            n_classes=self.n_classes,
            initial_class=int(y),
            support=1,
            birth_step=self.step,
            last_update_step=self.step,
        )
        self.rules.append(rule)
        self._event("create", rule_id=rid, y=int(y), reason=reason, best_activation=best_activation, n_rules=len(self.rules))

    def _update_one(self, x: np.ndarray, y: int) -> None:
        self.step += 1
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {x.size}")

        # Calculate assignment using knowledge available before seeing this label.
        w_before = self.feature_weights_
        if not self.rules:
            self._create_rule(x, y, reason="first_sample")
            self.sep_buffer.add_sample(x, y)
            self._add_similarity_sample(x)
            return

        best_idx, best_assignment = self._assign_sample_to_rule(x, w_before)
        best_regime = None
        best_unim = 0.0
        best_coverage = 0.0
        if best_idx is not None:
            rule0 = self.rules[best_idx]
            best_details = rule0.activation_details(x, w_before, record_regime=True)
            best_unim = float(best_details["activation"])
            best_coverage = float(self._rule_coverage(rule0, x, w_before))
            best_regime = best_details["regime"]
        if best_idx is None or best_assignment < self.alpha_add:
            self._create_rule(x, y, reason="low_coverage", best_activation=best_assignment)
            if self.event_log:
                self.event_log[-1]["best_regime"] = best_regime
                self.event_log[-1]["coverage"] = best_coverage
                self.event_log[-1]["unim_activation"] = best_unim
        else:
            rule = self.rules[best_idx]
            rid = rule.rule_id
            old_class = rule.dominant_class
            old_confidence = float(rule.class_probs[old_class])
            label_conflict = int(old_class) != int(y)
            if (self.label_conflict_policy == "new_rule" and label_conflict
                    and old_confidence >= self.label_conflict_min_confidence):
                # Stability/plasticity protection for supervised streams: a
                # well-established rule for the opposite class is not dragged
                # across the decision space merely because it covers x.
                self._create_rule(x, y, reason="label_conflict", best_activation=best_assignment)
                if self.event_log:
                    self.event_log[-1].update({
                        "conflicting_rule_id": int(rid),
                        "conflicting_rule_class": int(old_class),
                        "conflicting_rule_confidence": old_confidence,
                        "coverage": float(best_coverage),
                        "unim_activation": float(best_unim),
                        "best_regime": best_regime,
                    })
            else:
                rule.update_premise(x, self.step)
                rule.update_consequent(y)
                self._event("adapt", rule_id=rid, y=int(y), activation=float(best_unim), assignment_score=float(best_assignment), coverage=float(best_coverage), regime=best_regime, dominant_before=int(old_class), dominant_after=rule.dominant_class, support=rule.support)

        self.sep_buffer.add_sample(x, y)
        self._add_similarity_sample(x)
        w_after = self.feature_weights_
        self._maybe_merge_rules(w_after)
        self._enforce_max_rules()

    def _rule_coverage(self, rule: NeuroFuzzyRule, x: np.ndarray, feature_weights: np.ndarray) -> float:
        """Normalized membership coverage, independent of the Unim neutral element."""
        a = gaussian_memberships_per_feature(np.asarray(x, dtype=float), rule.center, rule.sigma)
        w = np.clip(np.asarray(feature_weights, dtype=float).reshape(-1), 0.0, 1.0)
        den = float(w.sum())
        return float(np.dot(w, a) / den) if den > 1e-12 else float(np.mean(a))

    def _rule_effective_activation(self, rule: NeuroFuzzyRule, x: np.ndarray, feature_weights: np.ndarray) -> float:
        u = float(rule.activation(x, feature_weights))
        if self.activation_mode == "coverage_gated_unim":
            return float(u * self._rule_coverage(rule, x, feature_weights))
        return u

    def _assign_sample_to_rule(self, x: np.ndarray, feature_weights: np.ndarray) -> Tuple[Optional[int], float]:
        if not self.rules:
            return None, 0.0
        if self.activation_mode == "coverage_gated_unim":
            acts = np.asarray([self._rule_coverage(r, x, feature_weights) for r in self.rules], dtype=float)
        else:
            acts = np.asarray([r.activation(x, feature_weights) for r in self.rules], dtype=float)
        idx = int(np.argmax(acts))
        return idx, float(acts[idx])

    def rule_coverages(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.rules:
            return np.zeros((X.shape[0], 0), dtype=float)
        w = self.feature_weights_
        return np.asarray([[self._rule_coverage(r, x, w) for r in self.rules] for x in X], dtype=float)

    def rule_activations(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.rules:
            return np.zeros((X.shape[0], 0), dtype=float)
        w = self.feature_weights_
        return np.asarray([[self._rule_effective_activation(r, x, w) for r in self.rules] for x in X], dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.rules:
            return np.ones((X.shape[0], self.n_classes), dtype=float) / self.n_classes
        A = self.rule_activations(X)
        class_mat = np.stack([r.class_probs for r in self.rules], axis=0)
        out = np.zeros((X.shape[0], self.n_classes), dtype=float)
        prior = np.sum([r.class_counts for r in self.rules], axis=0)
        prior = prior / prior.sum() if prior.sum() > 0 else np.ones(self.n_classes) / self.n_classes
        for i in range(X.shape[0]):
            s = A[i] @ class_mat
            out[i] = s / s.sum() if s.sum() > 1e-12 else prior
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def evidence(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Operational active-learning signals; must be empirically calibrated."""
        P = self.predict_proba(X)
        A = self.rule_activations(X)
        COV = self.rule_coverages(X) if self.activation_mode == "coverage_gated_unim" else A
        n = P.shape[0]
        novelty = np.ones(n, dtype=float) if COV.shape[1] == 0 else 1.0 - np.max(COV, axis=1)
        if self.n_classes == 1:
            uncertainty = np.zeros(n, dtype=float)
        else:
            ps = np.sort(P, axis=1)
            uncertainty = 1.0 - (ps[:, -1] - ps[:, -2])
        conflict = np.zeros(n, dtype=float)
        if self.rules:
            dom = np.asarray([r.dominant_class for r in self.rules], dtype=int)
            for i in range(n):
                total = float(A[i].sum())
                if total <= 1e-12:
                    conflict[i] = 0.0
                    continue
                class_activation = np.asarray([A[i, dom == c].sum() for c in range(self.n_classes)], dtype=float)
                conflict[i] = 1.0 - float(class_activation.max() / total)
                if self.n_classes > 1:
                    conflict[i] *= self.n_classes / (self.n_classes - 1.0)
        return {
            "proba": P,
            "novelty": np.clip(novelty, 0.0, 1.0),
            "uncertainty": np.clip(uncertainty, 0.0, 1.0),
            "conflict": np.clip(conflict, 0.0, 1.0),
            "max_activation": np.max(A, axis=1) if A.shape[1] else np.zeros(n, dtype=float),
            "max_coverage": 1.0 - np.clip(novelty, 0.0, 1.0),
        }

    def _compute_similarity_matrix(self, feature_weights: np.ndarray) -> Optional[np.ndarray]:
        r = len(self.rules)
        if r <= 1 or not self.sim_X_buffer:
            return None
        Xb = np.stack(self.sim_X_buffer, axis=0)
        A = np.asarray([[self._rule_effective_activation(rule, x, feature_weights) for x in Xb] for rule in self.rules], dtype=float)
        S = np.eye(r, dtype=float)
        for i in range(r):
            for j in range(i + 1, r):
                den = np.maximum(A[i], A[j]).sum()
                S[i, j] = S[j, i] = float(np.minimum(A[i], A[j]).sum() / den) if den > 1e-12 else 0.0
        return S

    def _compatible_consequents(self, ri: NeuroFuzzyRule, rj: NeuroFuzzyRule) -> Tuple[bool, float]:
        js = js_divergence(ri.class_probs, rj.class_probs)
        same = ri.dominant_class == rj.dominant_class
        ok = js <= self.merge_max_js and (same or not self.merge_require_same_class)
        return bool(ok), float(js)

    def _maybe_merge_rules(self, feature_weights: np.ndarray) -> None:
        # Merge at most one pair per observation for traceable evolution.
        # tau_merge > 1 is an explicit ablation/off switch; skip the expensive
        # functional-similarity matrix entirely rather than computing a matrix
        # that can never satisfy the threshold.
        if self.tau_merge > 1.0:
            return
        S = self._compute_similarity_matrix(feature_weights)
        if S is None:
            return
        np.fill_diagonal(S, -np.inf)
        candidates = []
        for i in range(S.shape[0]):
            for j in range(i + 1, S.shape[1]):
                if S[i, j] >= self.tau_merge:
                    ok, js = self._compatible_consequents(self.rules[i], self.rules[j])
                    if ok:
                        candidates.append((float(S[i, j]), -js, i, j, js))
        if not candidates:
            return
        _, _, i, j, js = max(candidates)
        ri, rj = self.rules[i], self.rules[j]
        ni, nj = float(ri.support), float(rj.support)
        n = ni + nj
        if n <= 0:
            return
        new_center = (ni * ri.center + nj * rj.center) / n
        # Pooled second moment: captures within-rule spread + between-center spread.
        var = (ni * (ri.sigma ** 2 + (ri.center - new_center) ** 2) + nj * (rj.sigma ** 2 + (rj.center - new_center) ** 2)) / n
        kept_id, removed_id = ri.rule_id, rj.rule_id
        ri.center = new_center
        ri.sigma = np.sqrt(np.maximum(var, 1e-6))
        ri.class_counts = ri.class_counts + rj.class_counts
        ri.support = int(ni + nj)
        ri.last_update_step = self.step
        for k, v in rj.regime_counts.items():
            ri.regime_counts[k] = ri.regime_counts.get(k, 0) + int(v)
        sim = float(S[i, j])
        del self.rules[j]
        self._event("merge", kept_rule_id=kept_id, removed_rule_id=removed_id, similarity=sim, js_divergence=float(js), n_rules=len(self.rules))

    def _enforce_max_rules(self) -> None:
        if self.max_rules is None or len(self.rules) <= self.max_rules:
            return
        idx = int(np.argmin([r.support for r in self.rules]))
        rid, support = self.rules[idx].rule_id, self.rules[idx].support
        del self.rules[idx]
        self._event("prune", rule_id=rid, support=int(support), reason="max_rules", n_rules=len(self.rules))

    def rule_snapshot(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for r in self.rules:
            probs = r.class_probs
            rows.append({
                "step": self.step,
                "rule_id": r.rule_id,
                "birth_step": r.birth_step,
                "last_update_step": r.last_update_step,
                "support": r.support,
                "dominant_class": r.dominant_class,
                "dominant_label": self.class_names[r.dominant_class],
                "confidence": float(probs.max()),
                "class_probs": probs.copy(),
                "regime_AND": int(r.regime_counts.get("AND", 0)),
                "regime_OR": int(r.regime_counts.get("OR", 0)),
                "regime_COMP": int(r.regime_counts.get("COMP", 0)),
            })
        return rows

    def explain_one(self, x: np.ndarray, top_k_rules: int = 3, top_k_features: int = 5) -> Dict[str, Any]:
        x = np.asarray(x, dtype=float).reshape(-1)
        w = self.feature_weights_
        P = self.predict_proba(x)[0]
        if not self.rules:
            return {"prediction": int(np.argmax(P)), "probabilities": P, "rules": []}
        details = []
        for r in self.rules:
            d = r.activation_details(x, w)
            coverage = float(self._rule_coverage(r, x, w))
            effective_activation = float(d["activation"] * coverage) if self.activation_mode == "coverage_gated_unim" else float(d["activation"])
            contribution = float(effective_activation * r.class_probs.max())
            feature_strength = w * d["membership"]
            idxs = np.argsort(feature_strength)[::-1][:top_k_features]
            features = [{
                "feature": self.feature_names[int(j)],
                "relevance": float(w[j]),
                "membership": float(d["membership"][j]),
                "mixed_input": float(d["mixed"][j]),
                "center": float(r.center[j]),
                "sigma": float(r.sigma[j]),
            } for j in idxs]
            details.append({
                "rule_id": r.rule_id,
                "activation": float(effective_activation),
                "unim_activation": float(d["activation"]),
                "coverage": coverage,
                "regime": d["regime"],
                "dominant_class": r.dominant_class,
                "dominant_label": self.class_names[r.dominant_class],
                "class_probs": r.class_probs.copy(),
                "support": r.support,
                "contribution": contribution,
                "top_features": features,
            })
        details.sort(key=lambda z: z["contribution"], reverse=True)
        return {"prediction": int(np.argmax(P)), "label": self.class_names[int(np.argmax(P))], "probabilities": P, "rules": details[:top_k_rules]}

    def rule_text(self, rule: NeuroFuzzyRule, top_k_features: int = 4) -> str:
        w = self.feature_weights_
        idxs = np.argsort(w)[::-1][:top_k_features]
        antecedents = [f"{self.feature_names[j]} near {rule.center[j]:.3f} (sigma {rule.sigma[j]:.3f}, relevance {w[j]:.2f})" for j in idxs]
        probs = rule.class_probs
        consequent = f"{self.class_names[int(np.argmax(probs))]} ({float(np.max(probs)):.2f})"
        return "IF " + " AND/COMP ".join(antecedents) + f" THEN {consequent}"
