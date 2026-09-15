"""Fixed-label QWK and thresholds fitted only on a designated validation set."""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

DEFAULT_THRESHOLDS = [0.5, 1.5, 2.5, 3.5, 4.5]


def grades(scores, thresholds=DEFAULT_THRESHOLDS):
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (5,) or not np.isfinite(thresholds).all() or not (np.diff(thresholds) > 0).all():
        raise ValueError('Provide five finite, strictly increasing thresholds')
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite predictions')
    # Right-inclusive bins match upstream pd.cut, including exact boundaries.
    return np.searchsorted(thresholds, scores, side='left')


def qwk(labels, predictions):
    if len(set(labels)) < 2 and len(set(predictions)) < 2 and np.array_equal(labels, predictions):
        return None  # Agreement is undefined for a single constant class.
    value = cohen_kappa_score(labels, predictions, labels=list(range(6)), weights='quadratic')
    return float(value) if np.isfinite(value) else None


def report(frame, thresholds=DEFAULT_THRESHOLDS):
    y = frame.isup_grade.to_numpy(dtype=int)
    pred = grades(frame.score, thresholds)
    result = {'samples': len(y), 'qwk': qwk(y, pred), 'accuracy': float(accuracy_score(y, pred)),
              'confusion_matrix': confusion_matrix(y, pred, labels=list(range(6))).tolist()}
    result['by_provider'] = {}
    for provider, group in frame.groupby('data_provider'):
        result['by_provider'][provider] = {'samples': len(group), 'qwk': qwk(group.isup_grade, grades(group.score, thresholds))}
    return result


def fit_thresholds(scores, labels):
    if len(set(labels)) < 2:
        raise ValueError('Threshold calibration requires at least two grades')
    def objective(values):
        if (np.diff(values) <= 0).any():
            return 2.0
        value = qwk(labels, grades(scores, values))
        return 1.0 if value is None else -value
    fitted = minimize(objective, np.array(DEFAULT_THRESHOLDS), method='Nelder-Mead', options={'maxiter': 1500})
    return fitted.x.tolist() if objective(fitted.x) < objective(np.array(DEFAULT_THRESHOLDS)) else DEFAULT_THRESHOLDS.copy()
