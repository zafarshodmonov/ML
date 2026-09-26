"""
ML5 — Supervised Learning: Decision Trees and Ensembles
=========================================================
"Don't Get Kicked" (Kaggle) uchun to'liq yechim.

Bu fayl quyidagilarni o'z ichiga oladi:
  1. Node, DecisionTreeClassifier, DecisionTreeRegressor  — noldan (from scratch), Gini/variance bilan
  2. Extra Randomized split (splitter="extra_random")      — ExtraTree uchun
  3. RandomForestClassifier                                 — bagging + feature subsampling
  4. ExtraTreesClassifier                                    — bagging'siz + extra randomized split
  5. GBDTClassifier                                          — binary cross-entropy gradient bilan, incremental
  6. gini_score, chronological train/valid/test split, kategorik encoding (unseen-safe)
  7. sklearn / LightGBM / XGBoost / CatBoost bilan solishtirish
  8. run_full_pipeline(csv_path) — haqiqiy "Don't Get Kicked" datasetida to'liq V-bob vazifasini bajaradi

Ishlatish:
  - Sinf implementatsiyalarini tekshirish uchun (haqiqiy dataset shart emas):
        python 03_solution.py --self-test
  - Haqiqiy Kaggle datasetida ishlatish uchun ("data/DontGetKicked.csv" yoki
    "data/training.csv" kabi CSV faylni data/ papkasiga qo'ying, README'dagi
    ko'rsatmaga muvofiq):
        python 03_solution.py --data data/training.csv --target IsBadBuy --date PurchDate
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# =====================================================================
# 0. Yordamchi metrika: Gini = 2*AUC - 1
# =====================================================================

def gini_score(y_true, y_proba) -> float:
    """Loyiha talab qilgan Gini metrikasi (ranking sifat o'lchovi, tugun impurity emas)."""
    auc = roc_auc_score(y_true, y_proba)
    return 2.0 * auc - 1.0


# =====================================================================
# 1. Node
# =====================================================================

@dataclass
class Node:
    """Decision tree tuguni. Ichki (internal) tugunda feature/threshold/children,
    barg (leaf) tugunda value (bashorat) saqlanadi."""
    is_leaf: bool = False
    value: Optional[np.ndarray] = None       # leaf: classifier uchun klass ehtimolliklari, regressor uchun skalyar
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    impurity: float = 0.0
    n_samples: int = 0

    @staticmethod
    def gini_impurity(y: np.ndarray, n_classes: int) -> float:
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=n_classes)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    @staticmethod
    def variance_impurity(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        return float(np.var(y))


# =====================================================================
# 2. Base Tree (Classifier va Regressor umumiy mantiqni ulashadi)
# =====================================================================

class _BaseTree:
    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,   # None -> barcha feature'lar ko'riladi
        splitter: str = "best",               # "best" (exhaustive CART) yoki "extra_random"
        random_state: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.splitter = splitter
        self.random_state = random_state
        self.root_: Optional[Node] = None
        self._rng = np.random.RandomState(random_state)

    # ---- impurity: subclass override qiladi ----
    def _impurity(self, y: np.ndarray) -> float:
        raise NotImplementedError

    def _leaf_value(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ---- best split qidirish (Gini/variance reduction'ni maksimallashtirish) ----
    #
    # "best" splitter uchun N ta threshold'ni bittalab tekshirish (har birida X<=t maskasi
    # va impurity'ni noldan hisoblash) — O(n^2) bo'lib ketadi. Buning o'rniga qiymatlarni
    # SARALAB, chap/o'ng statistikalarni (klass-bo'yicha kumulyativ yig'indilar) bitta
    # np.cumsum bilan hisoblaymiz — bu klassik "sorted cumulative sum" texnikasi, natijada
    # bitta feature uchun butun threshold qidiruvi O(n log n) (sort) ga tushadi, O(n^2) emas.
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)

        if self.max_features is None:
            feature_candidates = np.arange(n_features)
        else:
            k = min(self.max_features, n_features)
            feature_candidates = self._rng.choice(n_features, size=k, replace=False)

        best_gain = -np.inf
        best_feature, best_threshold = None, None

        for feature in feature_candidates:
            values = X[:, feature]

            if self.splitter == "extra_random":
                # Extra Randomized Tree: threshold TASODIFIY tanlanadi (min, max) oralig'ida, qidirilmaydi
                v_min, v_max = values.min(), values.max()
                if v_min == v_max:
                    continue
                threshold = self._rng.uniform(v_min, v_max)
                left_mask = values <= threshold
                n_left, n_right = int(left_mask.sum()), int((~left_mask).sum())
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                weighted = (n_left / n_samples) * self._impurity(y[left_mask]) + \
                           (n_right / n_samples) * self._impurity(y[~left_mask])
                gain = parent_impurity - weighted
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, int(feature), float(threshold)
                continue

            # "best" (exhaustive CART) — vektorlashtirilgan sorted-cumulative qidiruv
            gain, threshold = self._best_split_vectorized(values, y, n_samples, parent_impurity)
            if gain is not None and gain > best_gain:
                best_gain, best_feature, best_threshold = gain, int(feature), float(threshold)

        return best_feature, best_threshold, best_gain

    def _best_split_vectorized(self, values, y, n_samples, parent_impurity):
        raise NotImplementedError

    # ---- rekursiv tree qurish ----
    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(n_samples=len(y), impurity=self._impurity(y))

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or node.impurity <= 1e-12
        ):
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node

        feature, threshold, gain = self._find_best_split(X, y)

        if feature is None or gain <= 1e-12:
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node

        node.feature, node.threshold = feature, threshold
        left_mask = X[:, feature] <= threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._prepare(y)
        y_internal = self._transform_y(y)
        self.root_ = self._build(X, y_internal, depth=0)
        return self

    def _prepare(self, y):
        pass

    def _transform_y(self, y):
        return y

    def _predict_one(self, x: np.ndarray) -> np.ndarray:
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value


# =====================================================================
# 3. DecisionTreeClassifier
# =====================================================================

class DecisionTreeClassifier(_BaseTree):
    """Gini impurity kriteriysi bilan classification tree. Blueprint: fit / predict / predict_proba."""

    def _prepare(self, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        # y ni 0..K-1 ga map qilamiz (bincount uchun) — BIR MARTA, fit boshida
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}

    def _transform_y(self, y):
        # Vectorized mapping (np.searchsorted classes_ tartiblangan bo'lgani uchun ishlaydi)
        return np.searchsorted(self.classes_, y)

    def _impurity(self, y_idx: np.ndarray) -> float:
        return Node.gini_impurity(y_idx, self.n_classes_)

    def _leaf_value(self, y_idx: np.ndarray) -> np.ndarray:
        counts = np.bincount(y_idx, minlength=self.n_classes_)
        return counts / max(len(y_idx), 1)

    def _best_split_vectorized(self, values, y_idx, n_samples, parent_impurity):
        if n_samples < 2:
            return None, None
        order = np.argsort(values, kind="mergesort")
        sorted_vals = values[order]
        sorted_y = y_idx[order]

        onehot = np.zeros((n_samples, self.n_classes_))
        onehot[np.arange(n_samples), sorted_y] = 1.0
        cum = np.cumsum(onehot, axis=0)               # cum[i] = klass-bo'yicha soni birinchi (i+1) ta namunada
        total = cum[-1]

        left_counts = cum[:-1]                          # split "i" dan keyin: chapda i+1 ta namuna
        right_counts = total - left_counts
        n_left = np.arange(1, n_samples, dtype=float)
        n_right = n_samples - n_left

        valid = (sorted_vals[:-1] != sorted_vals[1:]) & \
                (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
        if not np.any(valid):
            return None, None

        gini_left = 1.0 - np.sum((left_counts / n_left[:, None]) ** 2, axis=1)
        gini_right = 1.0 - np.sum((right_counts / n_right[:, None]) ** 2, axis=1)
        weighted = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
        weighted = np.where(valid, weighted, np.inf)

        best_i = int(np.argmin(weighted))
        if not valid[best_i]:
            return None, None
        gain = parent_impurity - weighted[best_i]
        threshold = (sorted_vals[best_i] + sorted_vals[best_i + 1]) / 2.0
        return float(gain), float(threshold)

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# =====================================================================
# 4. DecisionTreeRegressor (MSE / variance)
# =====================================================================

class DecisionTreeRegressor(_BaseTree):
    """Standart og'ish (variance) kriteriysi bilan regression tree. GBDT ichida base learner sifatida ishlatiladi."""

    def _impurity(self, y: np.ndarray) -> float:
        return Node.variance_impurity(y)

    def _leaf_value(self, y: np.ndarray) -> np.ndarray:
        return np.array([float(np.mean(y))]) if len(y) else np.array([0.0])

    def _best_split_vectorized(self, values, y, n_samples, parent_impurity):
        if n_samples < 2:
            return None, None
        order = np.argsort(values, kind="mergesort")
        sorted_vals = values[order]
        sorted_y = y[order].astype(float)

        cum_sum = np.cumsum(sorted_y)
        cum_sq = np.cumsum(sorted_y ** 2)
        total_sum, total_sq = cum_sum[-1], cum_sq[-1]

        left_sum, left_sq = cum_sum[:-1], cum_sq[:-1]
        right_sum, right_sq = total_sum - left_sum, total_sq - left_sq
        n_left = np.arange(1, n_samples, dtype=float)
        n_right = n_samples - n_left

        valid = (sorted_vals[:-1] != sorted_vals[1:]) & \
                (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
        if not np.any(valid):
            return None, None

        # Var(t) = E[y^2] - E[y]^2
        var_left = left_sq / n_left - (left_sum / n_left) ** 2
        var_right = right_sq / n_right - (right_sum / n_right) ** 2
        weighted = (n_left / n_samples) * var_left + (n_right / n_samples) * var_right
        weighted = np.where(valid, weighted, np.inf)

        best_i = int(np.argmin(weighted))
        if not valid[best_i]:
            return None, None
        gain = parent_impurity - weighted[best_i]
        threshold = (sorted_vals[best_i] + sorted_vals[best_i + 1]) / 2.0
        return float(gain), float(threshold)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x)[0] for x in X])


# =====================================================================
# 5. RandomForestClassifier — bagging + feature subsampling
# =====================================================================

class RandomForestClassifier:
    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 8,
        max_features: Optional[int] = None,   # None -> sqrt(n_features) default sifatida fit() da tanlanadi
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees_: List[DecisionTreeClassifier] = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        max_features = self.max_features or max(1, int(math.sqrt(n_features)))
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)

        self.trees_ = []
        for b in range(self.n_trees):
            boot_idx = rng.randint(0, n_samples, size=n_samples)   # bootstrap: qaytarib qo'yish bilan
            X_b, y_b = X[boot_idx], y[boot_idx]
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=max_features,
                min_samples_leaf=self.min_samples_leaf,
                splitter="best",
                random_state=None if self.random_state is None else self.random_state + b,
            )
            tree.fit(X_b, y_b)
            self.trees_.append(tree)
        return self

    def predict_proba(self, X) -> np.ndarray:
        # Har bir tree o'z classes_ tartibida bashorat beradi; umumiy classes_ ga moslashtirib o'rtachalaymiz.
        all_probs = np.zeros((len(X), len(self.classes_)))
        for tree in self.trees_:
            p = tree.predict_proba(X)
            idx_map = [np.where(self.classes_ == c)[0][0] for c in tree.classes_]
            all_probs[:, idx_map] += p
        return all_probs / self.n_trees

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# =====================================================================
# 6. ExtraTreesClassifier — bagging YO'Q, extra randomized split BOR
# =====================================================================

class ExtraTreesClassifier(RandomForestClassifier):
    """RandomForest'dan farqi: (a) bootstrap qilinmaydi (butun dataset ishlatiladi,
    faqat feature'lar subsample qilinadi), (b) har bir split extra_random usulda tanlanadi."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        max_features = self.max_features or max(1, int(math.sqrt(n_features)))
        self.classes_ = np.unique(y)

        self.trees_ = []
        for b in range(self.n_trees):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=max_features,
                min_samples_leaf=self.min_samples_leaf,
                splitter="extra_random",       # <-- yagona farq: tasodifiy threshold
                random_state=None if self.random_state is None else self.random_state + b,
            )
            tree.fit(X, y)                     # <-- bootstrap YO'Q, to'liq X, y ishlatiladi
            self.trees_.append(tree)
        return self


# =====================================================================
# 7. GBDTClassifier — Binary Cross-Entropy gradient, incremental learning
# =====================================================================

class GBDTClassifier:
    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 3,
        max_features: Optional[int] = None,
        learning_rate: float = 0.1,
        subsample: float = 1.0,     # stochastic GBDT uchun qator subsampling (1.0 = o'chirilgan)
        random_state: Optional[int] = None,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.trees_: List[DecisionTreeRegressor] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        p0 = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.F0_ = math.log(p0 / (1 - p0))          # boshlang'ich log-odds
        F = np.full(n_samples, self.F0_)

        self.trees_ = []
        for m in range(self.n_trees):
            p = self._sigmoid(F)
            residuals = y - p                         # antigradient: BCE ning -dL/dF si

            if self.subsample < 1.0:
                sub_idx = rng.choice(n_samples, size=int(n_samples * self.subsample), replace=False)
            else:
                sub_idx = np.arange(n_samples)

            tree_m = DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=None if self.random_state is None else self.random_state + m,
            )
            tree_m.fit(X[sub_idx], residuals[sub_idx])

            update = tree_m.predict(X)
            F = F + self.learning_rate * update         # incremental: keyingi tree oldingisi ustiga qo'shiladi
            self.trees_.append(tree_m)
        return self

    def _raw_predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.F0_)
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X) -> np.ndarray:
        p1 = self._sigmoid(self._raw_predict(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# =====================================================================
# 8. Ma'lumot bilan ishlash: xronologik split + unseen-safe encoding
# =====================================================================

def chronological_split(df: pd.DataFrame, date_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train.PurchDate < valid.PurchDate < test.PurchDate, har biri ~33%."""
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    i1, i2 = n // 3, 2 * n // 3
    train = df_sorted.iloc[:i1]
    valid = df_sorted.iloc[i1:i2]
    test = df_sorted.iloc[i2:]
    return train, valid, test


class UnseenSafeLabelEncoder:
    """LabelEncoder'ning train'da ko'rilmagan kategoriyalarga (valid/test'da paydo bo'lgan)
    xavfsiz variant: bunday qiymatlar alohida "__UNSEEN__" kodiga tushadi, xato tashlamaydi."""

    def fit(self, series: pd.Series):
        self.categories_ = {v: i for i, v in enumerate(sorted(series.astype(str).unique()))}
        self.unseen_code_ = len(self.categories_)
        return self

    def transform(self, series: pd.Series) -> np.ndarray:
        return series.astype(str).map(lambda v: self.categories_.get(v, self.unseen_code_)).to_numpy()


def encode_categoricals(train, valid, test, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        enc = UnseenSafeLabelEncoder().fit(train[col])
        encoders[col] = enc
        for df in (train, valid, test):
            df[col] = enc.transform(df[col])
    return train, valid, test, encoders


# =====================================================================
# 9. To'liq pipeline (V-bob, 1-8 bandlar) — haqiqiy datasetda ishga tushiriladi
# =====================================================================

def run_full_pipeline(csv_path: str, target_col: str = "IsBadBuy", date_col: str = "PurchDate"):
    print(f"[1] Ma'lumot yuklanmoqda: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=[date_col])

    train, valid, test = chronological_split(df, date_col)
    print(f"    train={len(train)}  valid={len(valid)}  test={len(test)}")

    feature_cols = [c for c in df.columns if c not in (target_col, date_col)]
    categorical_cols = [c for c in feature_cols if train[c].dtype == object]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    for df_part in (train, valid, test):
        df_part[numeric_cols] = df_part[numeric_cols].fillna(df_part[numeric_cols].median())

    train, valid, test, _ = encode_categoricals(train, valid, test, categorical_cols)

    Xtr, ytr = train[feature_cols].to_numpy(float), train[target_col].to_numpy()
    Xva, yva = valid[feature_cols].to_numpy(float), valid[target_col].to_numpy()
    Xte, yte = test[feature_cols].to_numpy(float), test[target_col].to_numpy()

    results = {}

    # --- 2-3: o'zimizning DecisionTreeClassifier (max_depth talab qilingan parametr) ---
    print("[2] O'z DecisionTreeClassifier'imiz o'qitilmoqda ...")
    my_tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=20, random_state=42)
    my_tree.fit(Xtr, ytr)
    g = gini_score(yva, my_tree.predict_proba(Xva)[:, 1])
    print(f"    Gini(valid) = {g:.4f}  (talab: >= 0.10)")
    results["own_tree"] = g

    # --- 4: sklearn bilan solishtirish ---
    from sklearn.tree import DecisionTreeClassifier as SkTree
    sk_tree = SkTree(max_depth=7, min_samples_leaf=20, random_state=42)
    sk_tree.fit(Xtr, ytr)
    g = gini_score(yva, sk_tree.predict_proba(Xva)[:, 1])
    print(f"[4] sklearn DecisionTreeClassifier Gini(valid) = {g:.4f}")
    results["sklearn_tree"] = g

    # --- 5: Random Forest ---
    print("[5] RandomForestClassifier o'qitilmoqda ...")
    rf = RandomForestClassifier(n_trees=200, max_depth=8, random_state=42)
    rf.fit(Xtr, ytr)
    g = gini_score(yva, rf.predict_proba(Xva)[:, 1])
    print(f"    Gini(valid) = {g:.4f}  (talab: >= 0.15)")
    results["random_forest"] = g

    # --- 6: GBDT (o'zimizniki) ---
    print("[6] GBDTClassifier o'qitilmoqda ...")
    gbdt = GBDTClassifier(n_trees=150, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)
    gbdt.fit(Xtr, ytr)
    g = gini_score(yva, gbdt.predict_proba(Xva)[:, 1])
    print(f"    Gini(valid) = {g:.4f}")
    results["own_gbdt"] = g

    # --- 7: LightGBM / XGBoost / CatBoost (mavjud bo'lsa) ---
    best_lib_model, best_lib_name, best_lib_gini = None, None, -1.0
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1)
        m.fit(Xtr, ytr, categorical_feature=[feature_cols.index(c) for c in categorical_cols])
        g = gini_score(yva, m.predict_proba(Xva)[:, 1])
        print(f"[7] LightGBM Gini(valid) = {g:.4f}")
        results["lightgbm"] = g
        if g > best_lib_gini:
            best_lib_model, best_lib_name, best_lib_gini = m, "lightgbm", g
    except ImportError:
        print("[7] LightGBM o'rnatilmagan, o'tkazib yuborildi (pip install lightgbm)")

    try:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                               random_state=42, eval_metric="logloss")
        m.fit(Xtr, ytr)
        g = gini_score(yva, m.predict_proba(Xva)[:, 1])
        print(f"[7] XGBoost Gini(valid) = {g:.4f}")
        results["xgboost"] = g
        if g > best_lib_gini:
            best_lib_model, best_lib_name, best_lib_gini = m, "xgboost", g
    except ImportError:
        print("[7] XGBoost o'rnatilmagan, o'tkazib yuborildi (pip install xgboost)")

    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                                random_state=42, verbose=False)
        m.fit(Xtr, ytr)
        g = gini_score(yva, m.predict_proba(Xva)[:, 1])
        print(f"[7] CatBoost Gini(valid) = {g:.4f}")
        results["catboost"] = g
        if g > best_lib_gini:
            best_lib_model, best_lib_name, best_lib_gini = m, "catboost", g
    except ImportError:
        print("[7] CatBoost o'rnatilmagan, o'tkazib yuborildi (pip install catboost)")

    # --- 8: eng yaxshi modelni test'da baholash ---
    print("\n[8] Eng yaxshi modelni test datasetda baholash:")
    candidates = {"own_tree": (my_tree, "own"), "random_forest": (rf, "own"), "own_gbdt": (gbdt, "own")}
    if best_lib_model is not None:
        candidates[best_lib_name] = (best_lib_model, "lib")
    best_name = max(results, key=results.get) if results else None
    if best_name in candidates:
        model, kind = candidates[best_name]
        p_tr = model.predict_proba(Xtr)[:, 1]
        p_va = model.predict_proba(Xva)[:, 1]
        p_te = model.predict_proba(Xte)[:, 1]
        print(f"    Eng yaxshi model: {best_name}")
        print(f"    Gini(train) = {gini_score(ytr, p_tr):.4f}")
        print(f"    Gini(valid) = {gini_score(yva, p_va):.4f}")
        print(f"    Gini(test)  = {gini_score(yte, p_te):.4f}")

    # --- 9 (bonus): ExtraTrees ---
    print("\n[9*] ExtraTreesClassifier (bonus) o'qitilmoqda ...")
    et = ExtraTreesClassifier(n_trees=200, max_depth=8, random_state=42)
    et.fit(Xtr, ytr)
    g = gini_score(yva, et.predict_proba(Xva)[:, 1])
    print(f"    Gini(valid) = {g:.4f}  (talab: >= 0.12)")
    results["extra_trees"] = g

    return results


# =====================================================================
# 10. Self-test — sintetik ma'lumotda implementatsiyalarni tekshirish
#     (haqiqiy Kaggle dataseti hali yuklab olinmagan bo'lsa ham ishlaydi)
# =====================================================================

def self_test():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("=== SELF-TEST: sintetik dataset (make_classification) ===")
    X, y = make_classification(
        n_samples=4000, n_features=15, n_informative=8, n_redundant=2,
        weights=[0.85, 0.15], random_state=42,
    )
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    t0 = time.time()
    tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=10, random_state=42)
    tree.fit(Xtr, ytr)
    g = gini_score(yva, tree.predict_proba(Xva)[:, 1])
    print(f"DecisionTreeClassifier      Gini={g:.4f}   ({time.time()-t0:.2f}s)")
    assert g > 0.2, "Decision tree juda past sifat ko'rsatmoqda — kodda xato bo'lishi mumkin"

    t0 = time.time()
    reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    reg.fit(Xtr, ytr.astype(float))
    pred = reg.predict(Xva)
    mse = float(np.mean((pred - yva) ** 2))
    print(f"DecisionTreeRegressor       valid MSE={mse:.4f}   ({time.time()-t0:.2f}s)")

    t0 = time.time()
    rf = RandomForestClassifier(n_trees=50, max_depth=8, random_state=42)
    rf.fit(Xtr, ytr)
    g_rf = gini_score(yva, rf.predict_proba(Xva)[:, 1])
    print(f"RandomForestClassifier      Gini={g_rf:.4f}   ({time.time()-t0:.2f}s)")
    assert g_rf > g, "Random Forest bitta tree'dan yaxshiroq bo'lishi kutilgan edi"

    t0 = time.time()
    et = ExtraTreesClassifier(n_trees=50, max_depth=8, random_state=42)
    et.fit(Xtr, ytr)
    g_et = gini_score(yva, et.predict_proba(Xva)[:, 1])
    print(f"ExtraTreesClassifier        Gini={g_et:.4f}   ({time.time()-t0:.2f}s)")

    t0 = time.time()
    gbdt = GBDTClassifier(n_trees=100, max_depth=3, learning_rate=0.15, subsample=0.8, random_state=42)
    gbdt.fit(Xtr, ytr)
    g_gbdt = gini_score(yva, gbdt.predict_proba(Xva)[:, 1])
    print(f"GBDTClassifier (from scratch) Gini={g_gbdt:.4f}   ({time.time()-t0:.2f}s)")
    assert g_gbdt > g, "GBDT bitta tree'dan yaxshiroq bo'lishi kutilgan edi"

    from sklearn.tree import DecisionTreeClassifier as SkTree
    from sklearn.ensemble import RandomForestClassifier as SkRF, GradientBoostingClassifier as SkGB
    sk_tree = SkTree(max_depth=7, min_samples_leaf=10, random_state=42).fit(Xtr, ytr)
    sk_rf = SkRF(n_estimators=50, max_depth=8, random_state=42).fit(Xtr, ytr)
    sk_gb = SkGB(n_estimators=100, max_depth=3, learning_rate=0.15, random_state=42).fit(Xtr, ytr)
    print(f"[qiyoslash] sklearn Tree     Gini={gini_score(yva, sk_tree.predict_proba(Xva)[:,1]):.4f}")
    print(f"[qiyoslash] sklearn RF       Gini={gini_score(yva, sk_rf.predict_proba(Xva)[:,1]):.4f}")
    print(f"[qiyoslash] sklearn GBDT     Gini={gini_score(yva, sk_gb.predict_proba(Xva)[:,1]):.4f}")

    print("\nBarcha self-test tekshiruvlari muvaffaqiyatli o'tdi ✔")


# =====================================================================
# main
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML5: Decision Trees and Ensembles — solution")
    parser.add_argument("--self-test", action="store_true", help="Sintetik ma'lumotda implementatsiyani tekshirish")
    parser.add_argument("--data", type=str, default=None, help="Haqiqiy 'Don't Get Kicked' CSV fayli yo'li")
    parser.add_argument("--target", type=str, default="IsBadBuy", help="Target ustun nomi")
    parser.add_argument("--date", type=str, default="PurchDate", help="Sana ustuni nomi (xronologik split uchun)")
    args = parser.parse_args()

    if args.data:
        run_full_pipeline(args.data, target_col=args.target, date_col=args.date)
    else:
        # Default: hech qanday argument berilmasa ham, implementatsiya to'g'riligini tekshirish uchun self-test ishga tushadi.
        self_test()
