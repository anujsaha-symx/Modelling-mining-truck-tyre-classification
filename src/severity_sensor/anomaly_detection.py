"""
OSI Phase 3: Unsupervised Anomaly Detection
============================================
Fits Isolation Forest, Local Outlier Factor, and One-Class SVM
on derived features and produces a single Anomaly_Score ∈ [0, 100].
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# Columns used for anomaly detection (derived features only — no rolling window cols)
DERIVED_FEATURES = [
    'pressure_mean', 'pressure_std', 'pressure_drop_rate',
    'pressure_difference_between_tyres', 'pressure_zscore',
    'temperature_mean', 'temperature_std', 'temperature_max',
    'temperature_rise_rate', 'temperature_difference_between_tyres',
    'temperature_zscore',
    'payload_mean', 'payload_std', 'payload_change_rate', 'payload_percent_of_max',
    'speed_mean', 'speed_std', 'speed_max',
    'acceleration_magnitude', 'peak_acceleration', 'rms_vibration', 'crest_factor',
    'brake_frequency', 'brake_duration',
    'pitch_std', 'roll_std',
    'distance_increment', 'average_speed_from_gps',
]


def _safe_fit(estimator, X, label='estimator'):
    """Fit with error handling."""
    try:
        estimator.fit(X)
        return estimator
    except Exception as e:
        print(f'    [!] {label} fit failed: {e}')
        return None


def _safe_score(estimator, X, default=0.5, label='estimator'):
    """Score with error handling."""
    if estimator is None:
        return np.full(len(X), default)
    try:
        return estimator.score_samples(X)
    except Exception:
        try:
            return estimator.decision_function(X)
        except Exception as e:
            print(f'    [!] {label} scoring failed: {e}')
            return np.full(len(X), default)


def _normalise_anomaly(scores, lower_better: bool = True) -> np.ndarray:
    """Convert raw anomaly scores to [0, 100] where 100 = most anomalous.

    Parameters
    ----------
    scores : np.ndarray
        Raw scores. For IF/OCSVM, higher = more normal (lower_better=True).
    lower_better : bool
        True if lower raw score means more anomalous (IF/OCSVM).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not lower_better:
        scores = -scores
    lo, hi = np.nanmin(scores), np.nanmax(scores)
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo) * 100


class AnomalyEnsemble:
    """Ensemble of three unsupervised anomaly detectors.

    Fits on a reference DataFrame and transforms any DataFrame
    into Anomaly_Score ∈ [0, 100].
    """

    def __init__(self, n_jobs: int = -1, random_state: int = 42):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.iforest = None
        self.lof = None
        self.ocsvm = None
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        """Fit all three detectors on a reference sample.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns in DERIVED_FEATURES.
        """
        avail = [c for c in DERIVED_FEATURES if c in df.columns]
        if len(avail) < 3:
            raise ValueError(f'Need at least 3 derived features, got {len(avail)}')

        X = df[avail].fillna(df[avail].median()).values
        X_scaled = self.scaler.fit_transform(X)

        n = len(X_scaled)
        # Subsample for LOF (LOF is O(n²) in some impls; use 50k max)
        lof_n = min(n, 50000)
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(n, lof_n, replace=False)

        print(f'    Fitting IsolationForest on {n:,} samples ...')
        self.iforest = _safe_fit(
            IsolationForest(n_estimators=100, contamination='auto',
                            random_state=self.random_state, n_jobs=self.n_jobs),
            X_scaled, 'IsolationForest',
        )

        print(f'    Fitting LOF on {lof_n:,} samples ...')
        self.lof = _safe_fit(
            LocalOutlierFactor(n_neighbors=20, contamination='auto',
                               novelty=True, n_jobs=self.n_jobs),
            X_scaled[idx], 'LOF',
        )

        print(f'    Fitting OneClassSVM on 20k sample ...')
        svm_n = min(n, 20000)
        idx2 = rng.choice(n, svm_n, replace=False)
        self.ocsvm = _safe_fit(
            OneClassSVM(nu=0.05, kernel='rbf', gamma='scale'),
            X_scaled[idx2], 'OneClassSVM',
        )

        self._fitted = True
        print(f'    Anomaly detectors fitted on {len(avail)} features.')
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return Anomaly_Score ∈ [0, 100] for every row."""
        if not self._fitted:
            raise RuntimeError('Must fit before transform')

        avail = [c for c in DERIVED_FEATURES if c in df.columns]
        X = df[avail].fillna(df[avail].median()).values
        X_scaled = self.scaler.transform(X)

        s_if = _safe_score(self.iforest, X_scaled, default=0.0, label='IF')
        s_lof = _safe_score(self.lof, X_scaled, default=0.0, label='LOF')
        s_svm = _safe_score(self.ocsvm, X_scaled, default=0.0, label='OCSVM')

        a_if = _normalise_anomaly(s_if, lower_better=True)
        a_lof = _normalise_anomaly(s_lof, lower_better=False)
        a_svm = _normalise_anomaly(s_svm, lower_better=True)

        ensemble = (a_if + a_lof + a_svm) / 3.0
        result = pd.DataFrame({
            'timestamp': df['timestamp'].values,
            'Anomaly_Score': ensemble,
            'Anomaly_IF': a_if,
            'Anomaly_LOF': a_lof,
            'Anomaly_OCSVM': a_svm,
        })
        return result
