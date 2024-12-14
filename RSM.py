import numpy as np
import plotly.graph_objects as go

# Funkcja generująca front Pareto
def pareto_front(points):
    points = np.array(points)
    is_pareto = np.ones(points.shape[0], dtype=bool)  # Wszystkie punkty są kandydatami

    for i, point in enumerate(points):
        if is_pareto[i]:
            is_pareto[is_pareto] = ~np.all(points[is_pareto] <= point, axis=1) | np.all(points[is_pareto] == point, axis=1)
            is_pareto[i] = True  # Punkt pozostaje optymalny

    return points[is_pareto]

# Generowanie punktów w 4 wymiarach
def generate_elliptical_points_4d(center, axes, num_points, angle=30):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    r = np.random.uniform(0, 1, num_points)

    x = axes[0] * r * np.sin(phi) * np.cos(theta)
    y = axes[1] * r * np.sin(phi) * np.sin(theta)
    z = axes[2] * r * np.cos(phi)
    w = axes[3] * r

    angle_rad = np.radians(angle)
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)

    x_final = x_rot + center[0]
    y_final = y_rot + center[1]
    z_final = z + center[2]
    w_final = w + center[3]

    return np.column_stack((x_final, y_final, z_final, w_final))

# Wyznaczanie punktów referencyjnych
def calculate_reference_sets(all_data, pareto_depth=2):
    positive_reference = []
    all_data_list = list(all_data)

    for _ in range(pareto_depth):
        par = pareto_front(all_data_list)
        for p in par:
            positive_reference.append(p)
            all_data_list = [el for el in all_data_list if set(el) != set(p)]

    negative_reference = np.array(all_data_list)
    return np.array(positive_reference), negative_reference

# Algorytm RSM
def rsm_algorithm(pareto, positive_reference, negative_reference):
    scores = np.zeros(len(pareto))
    for i, p in enumerate(pareto):
        sp = min(np.linalg.norm(p - pr, ord=np.inf) for pr in positive_reference)
        sn = min(np.linalg.norm(p - nr, ord=np.inf) for nr in negative_reference)
        scores[i] = sn / (sp + sn)
    
    # Ranking punktów na podstawie wartości ocen
    ranking = np.argsort(-scores)  # Sortowanie malejąco
    return ranking, scores

# Funkcja wizualizująca wyniki
def visualize_rsm(positive_reference, data, negative_reference, pareto, best_point):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=positive_reference[:, 0], y=positive_reference[:, 1], z=positive_reference[:, 2],
        mode='markers', marker=dict(size=5, color='purple'), name="Punkty pozytywnej referencji"))
    fig.add_trace(go.Scatter3d(
        x=data[:, 0], y=data[:, 1], z=data[:, 2],
        mode='markers', marker=dict(size=5, color='blue'), name="Punkty docelowego zbioru"))
    fig.add_trace(go.Scatter3d(
        x=negative_reference[:, 0], y=negative_reference[:, 1], z=negative_reference[:, 2],
        mode='markers', marker=dict(size=5, color='green'), name="Punkty negatywnej referencji"))
    fig.add_trace(go.Scatter3d(
        x=pareto[:, 0], y=pareto[:, 1], z=pareto[:, 2],
        mode='markers', marker=dict(size=5, color=pareto[:, 3], colorscale='Viridis'),
        name="Punkty Pareto (kolor=F4)"))
    fig.add_trace(go.Scatter3d(
        x=[best_point[0]], y=[best_point[1]], z=[best_point[2]],
        mode='markers', marker=dict(size=10, color='red'), name="Najlepszy punkt"))

    fig.update_layout(
        title="RSM (3D z kolorem reprezentującym 4. wymiar)",
        scene=dict(xaxis_title="F1", yaxis_title="F2", zaxis_title="F3"),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1))
    
    fig.show(renderer="browser")

# Główna funkcja do uruchamiania analizy
def run_rsm_analysis():
    # Generowanie danych
    all_data = generate_elliptical_points_4d(center=(5, 10, 15, 20), axes=(4, 6, 8, 2), num_points=2000, angle=30)

    # Wyznaczanie punktów referencyjnych
    positive_reference, negative_reference = calculate_reference_sets(all_data)

    # Przetwarzanie danych i wyznaczanie frontu Pareto
    pareto = pareto_front(all_data)

    # Algorytm RSM
    ranking, scores = rsm_algorithm(pareto, positive_reference, negative_reference)

    # Najlepszy punkt
    best_point_index = ranking[0]
    best_point = pareto[best_point_index]

    # Wizualizacja
    visualize_rsm(positive_reference, all_data, negative_reference, pareto, best_point)

    # Zwracanie wyników
    return ranking, scores

# Uruchamianie analizy
ranking, scores = run_rsm_analysis()
print("Ranking punktów:", ranking)
print("Scores punktów:", scores)
