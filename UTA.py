import numpy as np
import plotly.graph_objects as go

# Funkcja definiująca zbiór niezdominowanych punktów
def pareto_front(points):
    points = np.array(points)
    is_pareto = np.ones(points.shape[0], dtype=bool)
    for i, point in enumerate(points):
        if is_pareto[i]:
            is_pareto[is_pareto] = ~np.all(points[is_pareto] <= point, axis=1) | np.all(points[is_pareto] == point, axis=1)
            is_pareto[i] = True
    return points[is_pareto]

# Funkcja klasyfikująca punkty referencyjne do FP(A*, θ) i FP(A*, -θ)
def classify_reference_points(A_star, theta):
    A_star = np.array(A_star)
    positive_reference = pareto_front(A_star + theta)
    negative_reference = pareto_front(A_star - theta)
    return positive_reference, negative_reference

# Funkcja użyteczności
def utility_function(point, positive_reference, negative_reference, lambda_, weights):
    sp = sum(weights[i] * min(abs(point[i] - pr[i]) for pr in positive_reference) for i in range(len(point)))
    sn = sum(weights[i] * min(abs(point[i] - nr[i]) for nr in negative_reference) for i in range(len(point)))
    sp_sn_sum = sp + sn or 1e-9  # Zapobieganie dzieleniu przez zero
    return lambda_ * sp / sp_sn_sum + (1 - lambda_) * sn / sp_sn_sum

# Funkcja UTA dla wariantu ciągłego
def uta_continuous(points, A_star, theta, lambda_=0.5, weights=None):
    points = np.array(points)
    if weights is None:
        weights = np.ones(points.shape[1]) / points.shape[1]

    positive_reference, negative_reference = classify_reference_points(A_star, theta)
    utility_values = np.array([utility_function(point, positive_reference, negative_reference, lambda_, weights) for point in points])
    best_solution_idx = np.argmax(utility_values)
    
    return best_solution_idx, utility_values

# Funkcja UTA dla wariantu dyskretnego
def uta_discrete(points, A_star, theta, lambda_=0.5, weights=None):
    points = np.array(points)
    if weights is None:
        weights = np.ones(points.shape[1]) / points.shape[1]

    positive_reference, negative_reference = classify_reference_points(A_star, theta)
    utility_values = np.array([utility_function(point, positive_reference, negative_reference, lambda_, weights) for point in points])
    best_solution_idx = np.argmax(utility_values)
    
    return best_solution_idx, utility_values

# Funkcja wizualizująca wyniki
def plot_4d_with_best(points, utilities, title="Ranking punktów w algorytmie UTA"):
    x, y, z, c = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    normalized_utilities = (utilities - np.min(utilities)) / (np.max(utilities) - np.min(utilities))
    best_idx = np.argmax(utilities)
    best_point = points[best_idx]
    best_utility = utilities[best_idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=6,
            color=normalized_utilities,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Funkcja<br>użyteczności")
        ),
        text=[f"u4={ci:.2f}, U={ui:.4f}" for ci, ui in zip(c, utilities)],
        name="Punkty"
    ))
    fig.add_trace(go.Scatter3d(
        x=[best_point[0]],
        y=[best_point[1]],
        z=[best_point[2]],
        mode='markers+text',
        marker=dict(
            size=10,
            color='red',
            symbol='diamond'
        ),
        text=[f"Najlepszy punkt<br>u4={best_point[3]:.2f}, U={best_utility:.4f}"],
        textposition="top center",
        name="Najlepszy punkt"
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Kryterium 1",
            yaxis_title="Kryterium 2",
            zaxis_title="Kryterium 3",
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show(renderer="browser")

# Przykład użycia
if __name__ == "__main__":
    # Generowanie danych
    np.random.seed(42)
    points = np.random.rand(100, 4)  # Punkty decyzyjne
    A_star = np.random.rand(10, 4)   # Punkty referencyjne
    theta = np.array([0.5, 0.5, 0.5, 0.5])
    lambda_ = 0.7
    weights = np.array([0.4, 0.3, 0.2, 0.1])

    # Wywołanie dla wariantu ciągłego
    best_idx_cont, utilities_cont = uta_continuous(points, A_star, theta, lambda_, weights)
    print(f"Najlepszy indeks (ciągły): {best_idx_cont}")
    print(f"Użyteczności (ciągły): {utilities_cont}")

    # Wywołanie dla wariantu dyskretnego
    best_idx_disc, utilities_disc = uta_discrete(points, A_star, theta, lambda_, weights)
    print(f"Najlepszy indeks (dyskretny): {best_idx_disc}")
    print(f"Użyteczności (dyskretny): {utilities_disc}")

    # Wizualizacja wyników
    plot_4d_with_best(points, utilities_cont)
