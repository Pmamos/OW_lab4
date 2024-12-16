import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib
matplotlib.use('TkAgg')

def fuzzy_topsis(
    alternatives,
    criteria,
    weights,
    variant="continuous",
    num_samples=0,
    bounds=None,
    metric="euclidean"
):
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

    def normalize_fuzzy_value(value, ideal):
        """Normalize a fuzzy value based on ideal solution."""
        return [value[k] / (ideal[k] or 1e-8) for k in range(3)]

    def calculate_fuzzy_distance(val, ref, metric="euclidean"):
        """Calculate fuzzy distance based on the chosen metric."""
        if metric == 'euclidean':
            return np.sqrt(sum((val[k] - ref[k]) ** 2 for k in range(3)))
        elif metric == 'chebyshev':
            return max(abs(val[k] - ref[k]) for k in range(3))
        raise ValueError("Unknown metric")

    if variant == "continuous" and num_samples > 0:
        alternatives, samples_mesh = generate_samples(alternatives, num_samples)
    else:
        samples_mesh = None

    num_alternatives, num_criteria = len(alternatives), len(criteria)

    ideal, anti_ideal = [], []
    for j in range(num_criteria):
        col = [alt[j] for alt in alternatives]
        if criteria[j]:
            ideal.append([max(c[k] for c in col) for k in range(3)])
            anti_ideal.append([min(c[k] for c in col) for k in range(3)])
        else:
            ideal.append([min(c[k] for c in col) for k in range(3)])
            anti_ideal.append([max(c[k] for c in col) for k in range(3)])

    normalized = [
        [normalize_fuzzy_value(alt[j], ideal[j]) for j in range(num_criteria)]
        for alt in alternatives
    ]

    weighted = [
        [[val[k] * weights[j][k] for k in range(3)] for j, val in enumerate(alt)]
        for alt in normalized
    ]

    distances_ideal = [
        sum(calculate_fuzzy_distance(val, ideal[j], metric) for j, val in enumerate(alt))
        for alt in weighted
    ]

    distances_anti_ideal = [
        sum(calculate_fuzzy_distance(val, anti_ideal[j], metric) for j, val in enumerate(alt))
        for alt in weighted
    ]

    closeness = [
        dist_anti / (dist_anti + dist_ideal)
        for dist_anti, dist_ideal in zip(distances_anti_ideal, distances_ideal)
    ]

    if variant == "continuous" and bounds is not None and num_samples > 0:
        closeness = []
        for point in samples_mesh:
            d_plus = min([calculate_fuzzy_distance(point, r_plus) for r_plus in ideal])
            d_minus = min([calculate_fuzzy_distance(point, r_minus) for r_minus in anti_ideal])
            closeness.append(d_minus / (d_minus + d_plus))

    ranking = np.argsort(closeness)[::-1]
    return ranking, {
        "normalized": normalized,
        "weighted": weighted,
        "distances_ideal": distances_ideal,
        "distances_anti_ideal": distances_anti_ideal,
        "closeness": closeness,
        "samples": samples_mesh
    }
