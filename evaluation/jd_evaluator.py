from typing import Any, Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


def flatten_nested_key(data: Dict[str, Any], field_path: str) -> List[Any]:
    """
    Retrieve a nested value from a dictionary using a dot-separated field path.

    If the result is a string with comma-separated values, it splits the string into a list of trimmed strings.
    If the result is a single value, it wraps it in a list.
    This function is used to prepare values for field-level evaluation.

    Args:
        data (Dict[str, Any]): A potentially nested dictionary of values.
        field_path (str): Dot-separated path to the desired field (e.g., "skills.hard_skills").

    Returns:
        List[Any]: A list of items extracted from the nested field.
    """
    keys = field_path.split(".")
    for key in keys:
        if isinstance(data, dict):
            if key in data:
                data = data[key]
            else:
                logger.warning(f"Field '{field_path}' not found in dictionary.")
                return []
        else:
            logger.warning(f"Field '{field_path}' not found: intermediate value is not a dictionary.")
            return []

    if isinstance(data, list):
        return data
    if isinstance(data, str):
        return [item.strip() for item in data.split(",")]
    return [data]


class FieldMetric:
    """
    Accumulates true positives, false positives, and false negatives to compute evaluation metrics.
    """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, true_items: List[str], pred_items: List[str]):
        """
        Update internal counts with new true and predicted items.

        Args:
            true_items (List[str]): List of true (ground truth) items.
            pred_items (List[str]): List of predicted items.
        """
        true_set = set(map(str.lower, true_items))
        pred_set = set(map(str.lower, pred_items))

        self.tp += len(true_set & pred_set)
        self.fp += len(pred_set - true_set)
        self.fn += len(true_set - pred_set)

    def compute(self) -> Dict[str, float]:
        """
        Compute and return precision, recall, F1 score, and support based on accumulated counts.

        Returns:
            Dict[str, float]: A dictionary with precision, recall, f1, and support values.
        """
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


class JDExtractionEvaluator:
    """
    Evaluate structured JD extraction using NER-style field-level metrics.
    """

    def __init__(self, fields: List[str]):
        """
        Args:
            fields (List[str]): List of dot-access fields to evaluate (e.g., skills.hard_skills)
        """
        self.fields = fields

    def evaluate_batch(self, ground_truths: List[Dict], predictions: List[Dict]):
        """
        Evaluate multiple job descriptions.

        Args:
            ground_truths (List[Dict]): Ground truth structured JD entries.
            predictions (List[Dict]): Model-extracted structured JD entries.

        Returns:
            Dict[str, dict]: Per-field metrics.
        """
        assert len(ground_truths) == len(predictions), "Mismatched number of ground truths and predictions"

        metrics = {field: FieldMetric() for field in self.fields}

        for gt, pred in zip(ground_truths, predictions):
            for field in self.fields:
                true_items = flatten_nested_key(gt, field)
                pred_items = flatten_nested_key(pred, field)
                metrics[field].update(true_items, pred_items)

        results = {field: metric.compute() for field, metric in metrics.items()}
        return results
