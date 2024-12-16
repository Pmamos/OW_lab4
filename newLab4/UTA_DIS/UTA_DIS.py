import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from typing import List, Tuple, Optional

def determine_criteria_bounds(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Determine the minimum and maximum values for each criterion."""
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    return min_values, max_values

def create_sample_grid(bounds: List[Tuple[float, float]], sample_count: int) -> np.ndarray:
    """
    Create a grid of samples within the given bounds with slight random variations.
    """
    samples = []
    for lower, upper in bounds:
        base_points = np.linspace(lower, upper, sample_count)
        random_noise = np.random.uniform(-0.1, 0.1, size=sample_count) * (upper - lower)
        samples.append(base_points + random_noise)
    return np.array(np.meshgrid(*samples)).T.reshape(-1, len(bounds))

def compute_partial_utilities(
    alternatives: np.ndarray,
    min_values: np.ndarray,
    max_values: np.ndarray,
    optimization_directions: List[bool],
    criteria_weights: List[float],
    is_continuous: bool = False,
    criteria_bounds: Optional[List[Tuple[float, float]]] = None,
    sample_count: int = 10
) -> np.ndarray:
    """Compute partial utilities for both discrete and continuous alternatives."""
    num_alternatives, num_criteria = alternatives.shape
    utilities = np.zeros((num_alternatives, num_criteria))

    for criterion_idx in range(num_criteria):
        value_range = max_values[criterion_idx] - min_values[criterion_idx]
        for alternative_idx in range(num_alternatives):
            if value_range == 0:
                normalized_value = 0.5  # If min == max, assume a middle value
            else:
                normalized_value = (alternatives[alternative_idx, criterion_idx] - min_values[criterion_idx]) / value_range

            if is_continuous:
                if criteria_bounds is None:
                    raise ValueError("Criteria bounds must be provided for continuous data.")
                criterion_samples = np.linspace(criteria_bounds[criterion_idx][0], criteria_bounds[criterion_idx][1], sample_count)
                normalized_value = np.interp(
                    alternatives[alternative_idx, criterion_idx],
                    criterion_samples,
                    np.linspace(0, 1, sample_count)
                )

            utilities[alternative_idx, criterion_idx] = (
                normalized_value * criteria_weights[criterion_idx]
                if optimization_directions[criterion_idx]
                else (1 - normalized_value) * criteria_weights[criterion_idx]
            )

    return utilities

def calculate_total_utilities(utilities: np.ndarray) -> np.ndarray:
    """Sum partial utilities to compute total utilities."""
    return np.sum(utilities, axis=1)

def assign_to_categories(total_utilities: np.ndarray, thresholds: List[float]) -> List[int]:
    """Assign utilities to categories based on thresholds."""
    categories = []
    for utility in total_utilities:
        for category_idx, threshold in enumerate(thresholds):
            if utility <= threshold:
                categories.append(category_idx + 1)
                break
        else:
            categories.append(len(thresholds) + 1)
    return categories

def uta_dis_algorithm(
    alternatives: np.ndarray,
    optimization_directions: List[bool],
    criteria_weights: Optional[List[float]] = None,
    thresholds: Optional[List[float]] = None,
    is_continuous: bool = False,
    criteria_bounds: Optional[List[Tuple[float, float]]] = None,
    sample_count: int = 10
) -> Tuple[Optional[List[int]], np.ndarray, np.ndarray]:
    """
    UTA-DIS algorithm for analyzing discrete and continuous alternatives.

    :param alternatives: Matrix of alternatives.
    :param optimization_directions: True for maximization, False for minimization.
    :param criteria_weights: Weights for each criterion (default=None, equal weights).
    :param thresholds: Thresholds for category classification.
    :param is_continuous: Whether the data represents continuous alternatives.
    :param criteria_bounds: Bounds for each criterion (required for continuous mode).
    :param sample_count: Number of samples for interpolation in continuous mode.
    :return: Categories, total utilities, and evaluated alternatives.
    """
    if is_continuous:
        if criteria_bounds is None:
            raise ValueError("Criteria bounds are required for continuous data.")
        alternatives = create_sample_grid(criteria_bounds, sample_count)

    num_criteria = alternatives.shape[1]
    if criteria_weights is None:
        criteria_weights = [1 / num_criteria] * num_criteria

    min_values, max_values = determine_criteria_bounds(alternatives)
    partial_utilities = compute_partial_utilities(
        alternatives, min_values, max_values, optimization_directions, criteria_weights, is_continuous, criteria_bounds, sample_count
    )

    total_utilities = calculate_total_utilities(partial_utilities)

    categories = assign_to_categories(total_utilities, thresholds) if thresholds is not None else None

    return categories, total_utilities, alternatives
