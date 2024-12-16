from typing import List, Tuple, Union
import numpy as np
from math import sqrt
import matplotlib
from scipy.spatial.distance import euclidean
matplotlib.use("TkAgg")

def distance(x: List[float], y: List[float]):
    return sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

def generate_samples(alternatives, num_samples):
    """Generate samples for continuous variant."""
    max_distance = 0
    point1, point2 = None, None
    for i in range(len(alternatives)):
        for j in range(i + 1, len(alternatives)):
            dist = sum(euclidean(alternatives[i][k], alternatives[j][k]) for k in range(len(alternatives[0])))
            if dist > max_distance:
                max_distance = dist
                point1, point2 = alternatives[i], alternatives[j]

    if point1 is not None and point2 is not None:
        samples = [
            np.linspace(point1[j][1], point2[j][1], num_samples)
            for j in range(len(point1))
        ]
        samples_mesh = np.array(np.meshgrid(*samples)).T.reshape(-1, len(point1))
        new_samples = [[(s - 1, s, s + 1) for s in sam] for sam in samples_mesh]
        return alternatives + new_samples, samples_mesh
    return alternatives, None

def find_pareto_front(
    decision_matrix: List[List[float]],
    min_max_criterial: List[bool]
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Wyznacza punkty niezdominowane i zdominowane w przestrzeni decyzji.

    Args:
        decision_matrix: Macierz decyzji (alternatywy i ich kryteria).
        min_max_criterial: Lista określająca, które kryteria są maksymalizowane.

    Returns:
        Tuple:
        - Lista punktów niezdominowanych.
        - Lista punktów zdominowanych.
    """
    matrix = np.array(decision_matrix, dtype=float).T

    for i, maximize in enumerate(min_max_criterial):
        if maximize:
            matrix[:, i] *= -1  # Odwróć wartości dla minimalizacji

    # Algorytm wyznaczania punktów Pareto
    is_pareto = np.ones(matrix.shape[0], dtype=bool)
    for i, point in enumerate(matrix):
        if is_pareto[i]:  # Tylko sprawdzaj, jeśli punkt jest jeszcze kandydatem
            # Sprawdź, które punkty są zdominowane przez point
            is_dominated = np.all(matrix <= point, axis=1) & np.any(matrix < point, axis=1)
            is_pareto[is_dominated] = False  # Odrzuć punkty zdominowane
            is_pareto[i] = True  # Aktualny punkt pozostaje niezdominowany
    pareto_points = matrix[is_pareto].tolist()
    dominated_points = matrix[~is_pareto].tolist()
    return pareto_points, dominated_points


def rsm_discrete(
    reference_points,
    decision_points,
    min_max: List[bool],
) -> List[Tuple[List, float, int]]:
    """
    Reference Set Method (RSM) w wariancie dyskretnym.
    :param reference_points: punkty referencyjne w przestrzeni kryteriów.
    :param decision_points: punkty decyzyjne
    :param min_max: Lista True (maksymalizacja) lub False (minimalizacja) dla każdego kryterium.
    :return: Zbiór punktów z obliczonymi odległościami do punktu referencyjnego.
    """
    ref_list = reference_points.tolist()
    decision_points_list = decision_points.tolist()
    pareto_positive, pareto_negative = find_pareto_front(ref_list, min_max)

    scores = []
    for point in decision_points_list:
        d_plus = min([distance(point, r_plus) for r_plus in pareto_positive])
        d_minus = min([distance(point, r_minus) for r_minus in pareto_negative])

        scores.append((point, d_minus - d_plus))

    scores.sort(key=lambda x: x[1], reverse=True)
    thresholds = [int(len(scores) * 0.1), int(len(scores) * 0.4)]

    for idx, (point, score) in enumerate(scores):
        category = 1 if idx <= thresholds[0] else 2 if idx <= thresholds[1] else 3
        scores[idx] = (point, score, category)

    return scores


def rsm_continuous(
    num_samples: int,
    reference_points,
    min_max: List[bool],
    bounds: Union[List[Tuple[float, float]], float] = 0,
) -> List[Tuple[List, float, int]]:
    """
    Reference Set Method (RSM) w wariancie ciągłym.
    :param num_samples: liczba próbek.
    :param bounds: lista krotek (min, max) dla każdego kryterium.
    :param min_max: Lista True (maksymalizacja) lub False (minimalizacja) dla każdego kryterium.
    :param reference_points: punkt referencyjny w przestrzeni kryteriów.
    :return: Zbiór punktów z obliczonymi odległościami do punktu referencyjnego.
    """
    if bounds == 0:
        bounds = [(0, 10) for _ in range(len(min_max))]

    ref_list = reference_points.tolist()
    sample_ranges = [np.linspace(b[0], b[1], num_samples) for b in bounds]
    samples_mesh = np.array(np.meshgrid(*sample_ranges)).T.reshape(-1, len(bounds))

    # Wyznaczenie punktów Pareto
    pareto_positive, pareto_negative = find_pareto_front(ref_list, min_max)
    print(pareto_negative)

    # Obliczanie odległości od punktów referencyjnych
    scores = []
    for point in samples_mesh:
        d_positive = min(distance(point, ref) for ref in pareto_positive)
        d_negative = min(distance(point, ref) for ref in pareto_negative)
        scores.append((point.tolist(), d_negative - d_positive))

    # Sortowanie i przypisywanie kategorii
    scores.sort(key=lambda x: x[1], reverse=True)
    thresholds = [int(len(scores) * 0.1), int(len(scores) * 0.4)]

    for idx, (point, score) in enumerate(scores):
        category = 1 if idx <= thresholds[0] else 2 if idx <= thresholds[1] else 3
        scores[idx] = (point, score, category)

    return scores
