"""
Empathy Package (prototype)

Contains:
- EmpathyScorer: computes empathy components (global mean, pc1 with optional prior, knn affinity) and returns empathy scores.
- EPINNModel: lightweight experimental E-PINN wrapper. Provides fit/predict skeleton and integrates simple physics-informed penalties.

This is a research prototype for experimentation and integration with the README. It is intentionally simple and readable.

Usage example:

from empathy_package import EmpathyScorer, EPINNModel

# build toy dataset (list-like columns or pandas-like dict)
data = {
    'visits': [980,1020,880,1100,970],
    'stories': [3,4,2,5,3],
    'clicks': [200,220,190,250,205],
    'moon_phase': [2,3,4,0,1],
    'sales': [38,42,33,48,37]
}

scorer = EmpathyScorer()
empathy = scorer.calculate_empathy(data)

model = EPINNModel()
model.fit(data, empathy)
print(model.predict(data))

Note: this is a prototype. For production use, add robust validation, batching, device handling (GPU), and tests.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import math


class EmpathyScorer:
    """Compute empathy scores from tabular-like data.

    Components implemented:
    - global mean alignment
    - principal component (pc1) projection with optional prior
    - local kNN cosine affinity (approximate)

    Input: dictionary with column-name -> list/array (equal lengths)
    Output: numpy array of empathy Z-scores
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, k: int = 3):
        # default combination weights
        if weights is None:
            weights = {'global': 0.4, 'pc1': 0.4, 'knn': 0.2}
        self.w = weights
        self.k = k

    @staticmethod
    def _to_matrix(data: Dict[str, List[float]], feature_order: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        # convert dict to numpy matrix (n_samples, n_features)
        keys = list(data.keys()) if feature_order is None else feature_order
        arrs = [np.asarray(data[k], dtype=float) for k in keys]
        n = arrs[0].shape[0]
        mat = np.vstack([a.reshape(n) for a in arrs]).T
        return mat, keys

    @staticmethod
    def _zscore(X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, keepdims=True) + 1e-12
        return (X - mu) / sigma

    def _global_mean(self, Xz: np.ndarray) -> np.ndarray:
        mu = Xz.mean(axis=0, keepdims=True)
        mu_norm = np.linalg.norm(mu) + 1e-12
        mu_dir = mu / mu_norm
        return (Xz @ mu_dir.T).ravel()

    def _pc1(self, Xz: np.ndarray, init_v: Optional[np.ndarray] = None) -> np.ndarray:
        # PCA via SVD
        U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
        v1 = Vt[0]
        if init_v is not None:
            iv = np.asarray(init_v, dtype=float)
            iv = iv / (np.linalg.norm(iv) + 1e-12)
            v1 = 0.7 * v1 + 0.3 * iv
        v1 = v1 / (np.linalg.norm(v1) + 1e-12)
        return Xz @ v1

    def _knn_affinity(self, Xz: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        if k is None:
            k = self.k
        norms = np.linalg.norm(Xz, axis=1, keepdims=True) + 1e-12
        Xn = Xz / norms
        S = Xn @ Xn.T
        np.fill_diagonal(S, -np.inf)
        # for very small n, cap k
        k = min(k, max(1, Xz.shape[0] - 1))
        idx = np.argpartition(S, -k, axis=1)[:, -k:]
        topk = S[np.arange(S.shape[0])[:, None], idx]
        return topk.mean(axis=1)

    def calculate_empathy(self, data: Dict[str, List[float]], feature_order: Optional[List[str]] = None,
                          init_v: Optional[List[float]] = None) -> np.ndarray:
        """Main entry. Returns empathy z-scores (mean 0, std 1) for each sample."""
        X, keys = self._to_matrix(data, feature_order)
        Xz = self._zscore(X)
        g = self._global_mean(Xz)
        p = self._pc1(Xz, init_v=init_v)
        k = self._knn_affinity(Xz, k=self.k)
        combined = self.w['global'] * g + self.w['pc1'] * p + self.w['knn'] * k
        # normalize to z-score
        comb_z = (combined - combined.mean()) / (combined.std() + 1e-12)
        return comb_z


class EPINNModel:
    """A minimal Empathetic Physics-Informed Neural Network wrapper.

    This is a small experimental model that demonstrates how the empathy feature can be
    integrated into a predictive workflow and how simple physics-informed penalties can
    be added to the loss.

    The implementation below uses a linear model + optional physics penalties for transparency.
    For deeper experiments, substitute with a TF/PyTorch MLP and batch-based penalties.
    """

    def __init__(self, lambda_sat: float = 1e-2, lambda_cons: float = 1e-2, lambda_lat: float = 1e-2):
        self.lambda_sat = float(lambda_sat)
        self.lambda_cons = float(lambda_cons)
        self.lambda_lat = float(lambda_lat)
        self.is_fitted = False
        # model params
        self.coef_ = None
        self.bias_ = None

    @staticmethod
    def _build_design_matrix(data: Dict[str, List[float]], empathy_z: np.ndarray,
                             feature_order: Optional[List[str]] = None) -> np.ndarray:
        X, keys = EmpathyScorer._to_matrix(data, feature_order)
        # z-score numeric columns
        Xz = EmpathyScorer._zscore(X)
        return np.column_stack([Xz, empathy_z.reshape(-1, 1)])

    def fit(self, data: Dict[str, List[float]], empathy_z: np.ndarray, target: Optional[List[float]] = None,
            feature_order: Optional[List[str]] = None, epochs: int = 2000, lr: float = 1e-2):
        """Fit a simple linear model with physics-informed penalties.

        If target is None, EPINN will attempt to reconstruct 'sales' if present in data.
        """
        n = len(empathy_z)
        X = self._build_design_matrix(data, empathy_z, feature_order)
        if target is None:
            if 'sales' in data:
                y = np.asarray(data['sales'], dtype=float).reshape(-1, 1)
            else:
                raise ValueError("No target provided and 'sales' not found in data")
        else:
            y = np.asarray(target, dtype=float).reshape(-1, 1)

        # initialize params
        rng = np.random.RandomState(42)
        d = X.shape[1]
        w = rng.normal(scale=0.1, size=(d, 1))
        b = np.array([[0.0]])

        observed_total = y.sum()
        total_slack = 0.25 * observed_total

        # simple gradient descent (linear model) with physics penalties
        for epoch in range(epochs):
            preds = X @ w + b
            mse = np.mean((preds - y) ** 2)
            # saturation proxy: finite diff on visits (col 0) and clicks (col 2)
            eps = 1e-2
            X_up = X.copy(); X_dn = X.copy()
            if X.shape[1] >= 3:
                X_up[:, 0] += eps; X_dn[:, 0] -= eps
                p_up = X_up @ w + b; p_dn = X_dn @ w + b
                sec_vis = p_up - 2 * preds + p_dn
                sat_vis = np.mean(np.maximum(sec_vis, 0.0))

                X_up2 = X.copy(); X_dn2 = X.copy()
                X_up2[:, 2] += eps; X_dn2[:, 2] -= eps
                p_up2 = X_up2 @ w + b; p_dn2 = X_dn2 @ w + b
                sec_click = p_up2 - 2 * preds + p_dn2
                sat_click = np.mean(np.maximum(sec_click, 0.0))
            else:
                sat_vis = 0.0; sat_click = 0.0

            sat_pen = float(sat_vis + sat_click)
            total_pred = float(np.sum(preds))
            cons_pen = max(0.0, (total_pred - (observed_total + total_slack)) / (observed_total + 1e-12))
            preds_r = preds.flatten()
            if len(preds_r) > 1:
                diffs = preds_r[1:] - preds_r[:-1]
                lat_pen = float(np.mean(diffs ** 2))
            else:
                lat_pen = 0.0

            loss = mse + self.lambda_sat * sat_pen + self.lambda_cons * cons_pen + self.lambda_lat * lat_pen

            # gradients (analytic for linear model)
            grad_w = (2.0 / n) * (X.T @ (preds - y))
            grad_b = (2.0 / n) * np.sum(preds - y)

            if cons_pen > 0:
                grad_w += (1.0 / (observed_total + 1e-12)) * np.sum(X, axis=0).reshape(-1, 1)
                grad_b += (1.0 / (observed_total + 1e-12)) * n

            # latency grad approx
            if len(preds_r) > 1:
                Xdiff = X[1:, :] - X[:-1, :]
                pdiff = preds_r[1:] - preds_r[:-1]
                grad_lat_w = (2.0 / (n - 1)) * (Xdiff.T @ pdiff.reshape(-1, 1))
                grad_w += self.lambda_lat * grad_lat_w

            # update
            w -= lr * grad_w
            b -= lr * grad_b

        self.coef_ = w
        self.bias_ = b
        self.is_fitted = True

    def predict(self, data: Dict[str, List[float]], empathy_z: np.ndarray, feature_order: Optional[List[str]] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X = self._build_design_matrix(data, empathy_z, feature_order)
        preds = X @ self.coef_ + self.bias_
        return preds.ravel()

    def generate_visitor_ids(self, empathy_z: np.ndarray, n_classes: int = 24) -> List[str]:
        """Generate simple visitor IDs in the form score&class using KMeans on empathy+features.

        Note: This is a light utility intended for examples. For production, use a persistent
        clustering and stable class mapping.
        """
        from sklearn.cluster import KMeans
        # this method requires the original data matrix be accessible externally; for demo we accept empathy_z only
        # naive mapping: quantile score from empathy + random class assignment
        qs = np.quantile(empathy_z, [0.2, 0.4, 0.6, 0.8])
        scores = np.digitize(empathy_z, qs) + 1  # 1..5
        rng = np.random.RandomState(42)
        classes = rng.randint(0, n_classes, size=empathy_z.shape[0])
        ids = [f"{int(s)}&{int(c)}" for s, c in zip(scores, classes)]
        return ids


# end of empathy_package.py
