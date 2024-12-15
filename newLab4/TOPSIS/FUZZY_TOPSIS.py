import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from scipy.spatial.distance import euclidean

def fuzzy_topsis(
    alternatives,
    criteria,
    weights,
    variant="continuous",
    num_samples=0,
    bounds=None,
    metric="euclidean"
):
    """
    Fuzzy TOPSIS Algorithm.
    :param alternatives: Array of alternatives, each represented as an array of triangular fuzzy numbers.
    :param criteria: Type of each criterion ('benefit' or 'cost').
    :param weights: Array of weights, each represented as a triangular fuzzy number.
    :param variant: Type of Fuzzy TOPSIS ('continuous' or 'discrete').
    :param num_samples: Number of samples for continuous variant (if applicable).
    :param bounds: List of tuples (min, max) for each criterion (if applicable).
    :return: Ranking of alternatives and intermediate matrices.
    """



    samples = None
    if variant == "continuous" and num_samples > 0:
        # Skorzystaj z istniejących alternatyw dyskretnych jako punktów startowych
        discrete_alternatives = alternatives.copy()

        # Znajdź dwa punkty, które są najdalej od siebie
        max_distance = 0
        point1, point2 = None, None
        for i in range(len(discrete_alternatives)):
            for j in range(i + 1, len(discrete_alternatives)):
                dist = sum(euclidean(discrete_alternatives[i][k], discrete_alternatives[j][k]) for k in
                           range(len(discrete_alternatives[0])))
                if dist > max_distance:
                    max_distance = dist
                    point1, point2 = discrete_alternatives[i], discrete_alternatives[j]

        # Generowanie nowych próbek między punktami
        if point1 is not None and point2 is not None:
            samples = [
                np.linspace(point1[j][1], point2[j][1], num_samples)
                for j in range(len(point1))
            ]

            samples_mesh = np.array(np.meshgrid(*samples)).T.reshape(-1, len(point1)).tolist()
            new_alternatives = [[(s - 1, s, s + 1) for s in sam] for sam in samples_mesh]
            new_samples = [[(s - 1, s, s + 1) for s in sam] for sam in samples_mesh]

            # Połącz istniejące alternatywy dyskretne i nowe próbki ciągłe
            alternatives = discrete_alternatives + new_alternatives
            samples = discrete_alternatives + new_samples
        else:
            samples = discrete_alternatives

    num_alternatives, num_criteria = len(alternatives), len(criteria)

    def normalize_fuzzy(value, ideal):
        """Normalize a fuzzy value based on ideal solution."""
        try:
            v1 = (value[0] / ideal[0])
        except:
            v1 = (value[0] / 0.00000001)

        try:
            v2 = (value[1] / ideal[1])
        except:
            v2 = (value[1] / 0.00000001)

        try:
            v3 = (value[2] / ideal[2])
        except:
            v3 = (value[2] / 0.00000001)
        return [v1, v2, v3]

    # Calculate fuzzy ideal and anti-ideal solutions
    ideal = []
    anti_ideal = []
    for j in range(num_criteria):
        col = [alt[j] for alt in alternatives]
        if criteria[j] is True:
            ideal.append([max(c[0] for c in col), max(c[1] for c in col), max(c[2] for c in col)])
            anti_ideal.append([min(c[0] for c in col), min(c[1] for c in col), min(c[2] for c in col)])
        elif criteria[j] is False:
            ideal.append([min(c[0] for c in col), min(c[1] for c in col), min(c[2] for c in col)])
            anti_ideal.append([max(c[0] for c in col), max(c[1] for c in col), max(c[2] for c in col)])



    # Normalize alternatives
    normalized = [
        [normalize_fuzzy(alt[j], ideal[j]) for j in range(num_criteria)]
        for alt in alternatives
    ]

    print(normalized)

    # Weighted normalized fuzzy decision matrix
    weighted = [
        [[val[k] * weights[j][k] for k in range(3)] for j, val in enumerate(alt)]
        for alt in normalized
    ]

    # Distance to fuzzy ideal and anti-ideal solutions
    def fuzzy_distance(val, ref, met="euclidean"):
        """Calculate fuzzy distance."""
        if met == 'euclidean':
            return np.sqrt(sum((val[k] - ref[k]) ** 2 for k in range(3)))
        elif met == 'chebyshev':
            return max((abs(val[k] - ref[k]) ** 2 for k in range(3)))
        else:
            raise ValueError("Unknown metric")

    distances_ideal = [sum(fuzzy_distance(val, ideal[j], metric) for j, val in enumerate(alt)) for alt in weighted]

    distances_anti_ideal = [sum(fuzzy_distance(val, anti_ideal[j], metric) for j, val in enumerate(alt)) for alt in weighted]

    # Calculate closeness coefficient
    closeness = [dist_anti / (dist_anti + dist_ideal) for dist_anti, dist_ideal in zip(distances_anti_ideal, distances_ideal)]
    # samples = None
    if variant == "continuous" and bounds is not None and num_samples > 0:
        continuous_scores = []
        for point in samples_mesh:
            d_plus = min([fuzzy_distance(point, r_plus) for r_plus in ideal])
            d_minus = min([fuzzy_distance(point, r_minus) for r_minus in anti_ideal])
            continuous_scores.append(d_minus / (d_minus + d_plus))

        closeness = continuous_scores
        samples = [[(s-1, s, s+1) for s in sam] for sam in samples_mesh]

    # Return ranking for the discrete variant
    ranking = np.argsort(closeness)[::-1]
    return ranking, {
        "normalized": normalized,
        "weighted": weighted,
        "distances_ideal": distances_ideal,
        "distances_anti_ideal": distances_anti_ideal,
        "closeness": closeness,
        "samples": samples
    }

