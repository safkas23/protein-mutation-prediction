import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import ttest_rel


def evaluate_regression_model(y_true, y_pred):
    """
    Compute regression metrics.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


def paired_t_test(model_a_scores, model_b_scores):
    """
    Statistical comparison between two models
    across CV folds.
    """
    stat, p_value = ttest_rel(model_a_scores, model_b_scores)
    return {
        "t_stat": stat,
        "p_value": p_value
    }


def compute_cv_summary(scores):
    """
    Convert CV scores into mean ± std format.
    """
    return {
        "mean": np.mean(scores),
        "std": np.std(scores)
    }