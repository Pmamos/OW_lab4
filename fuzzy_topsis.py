import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.decomposition import PCA

class FuzzyNumber:
    def __init__(self, l, m, u):
        self.l = l  # Lower bound
        self.m = m  # Modal value
        self.u = u  # Upper bound

    def __repr__(self):
        return f"({self.l}, {self.m}, {self.u})"
    
    def distance(self, other) -> float:
        return np.sqrt(1/3 * ((self.l - other.l)**2 + (self.m - other.m)**2 + (self.u - other.u)**2))
     
    def normalize(self, divisor, w):
        return FuzzyNumber(self.l * w / divisor, self.m * w / divisor, self.u * w / divisor)

def objectives(u, p1=1, p2=1, p3=1, p4=1):
    x, y, z, w = u
    f1_med = x**2 + y**2 + z**2 + w**2
    f2_med = (x - p1)**2 + (y + p2)**2 + (z - p3)**2 + (w + p4)**2
    f3_med = (x + p1)**2 + (y - p2)**2 + (z + p3)**2 + (w - p4)**2
    f4_med = (x - p1)**2 + (y - p2)**2 + (z + p3)**2 + (w + p4)**2
    return [
        FuzzyNumber(f1_med - 2, f1_med, f1_med + 2),
        FuzzyNumber(f2_med - 2, f2_med, f2_med + 2),
        FuzzyNumber(f3_med - 2, f3_med, f3_med + 2),
        FuzzyNumber(f4_med - 2, f4_med, f4_med + 2),
    ]

def normalize_with_weights(data, w, criteria):
    norm_matrix = np.zeros_like(data, dtype=object)
    for j in range(len(data[0])):
        column = [row[j] for row in data]
        if criteria[j] == 'max':
            max_value = max(f.u for f in column)
            divisor = max_value
        else:
            min_value = min(f.l for f in column)
            divisor = min_value
        
        norm_matrix[:, j] = [f.normalize(divisor, w[j]) for f in column]
    
    return norm_matrix

def fuzzy_topsis(data, weights, criteria):
    # Normalization of data with weights
    normalized_data = normalize_with_weights(data, weights, criteria)

    # Determine the ideal best and worst points
    a_pos = [FuzzyNumber(min([fn.l for fn in col]), min([fn.m for fn in col]), min([fn.u for fn in col])) for col in normalized_data.T]
    a_neg = [FuzzyNumber(max([fn.l for fn in col]), max([fn.m for fn in col]), max([fn.u for fn in col])) for col in normalized_data.T]

    # Calculate distances and ranking
    cs = []
    for row in normalized_data:
        sp = np.sqrt(sum([f.distance(a_pos[i])**2 for i, f in enumerate(row)]))
        sn = np.sqrt(sum([f.distance(a_neg[i])**2 for i, f in enumerate(row)]))
        cs.append(sn / (sp + sn))

    # Ranking points
    ranking_indices = np.argsort(-np.array(cs))
    ranked_data = [data[i] for i in ranking_indices]
    ranked_scores = [cs[i] for i in ranking_indices]

    return ranked_data, ranked_scores

import numpy as np

def generate_elliptical_points_4d(center, axes, num_points, angles=(30, 45)):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    r = np.random.uniform(0, 1, num_points)

    # Transform into ellipsoidal space
    x = axes[0] * r * np.sin(phi) * np.cos(theta)
    y = axes[1] * r * np.sin(phi) * np.sin(theta)
    z = axes[2] * r * np.cos(phi)
    w = axes[3] * r

    # Apply rotation in 2D space (x-y and z-w)
    angle_rad_xy = np.radians(angles[0])
    angle_rad_zw = np.radians(angles[1])

    x_rot = x * np.cos(angle_rad_xy) - y * np.sin(angle_rad_xy)
    y_rot = x * np.sin(angle_rad_xy) + y * np.cos(angle_rad_xy)
    z_rot = z * np.cos(angle_rad_zw) - w * np.sin(angle_rad_zw)
    w_rot = z * np.sin(angle_rad_zw) + w * np.cos(angle_rad_zw)

    # Translate to the center of the ellipse
    x_final = x_rot + center[0]
    y_final = y_rot + center[1]
    z_final = z_rot + center[2]
    w_final = w_rot + center[3]

    return np.column_stack((x_final, y_final, z_final, w_final))

# Example usage for continuous data
x_vals = np.linspace(-10, 10, 10)
y_vals = np.linspace(-1, 1, 10)
z_vals = np.linspace(-5, 5, 10)
w_vals = np.linspace(0, 20, 10)

X, Y, Z, W = np.meshgrid(x_vals, y_vals, z_vals, w_vals)
U_grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), W.ravel()])
data_continuous = [objectives(p, 1, 1) for p in U_grid]
criteria = ['max', 'min', 'max', 'min']
weights = np.array([0.4, 0.3, 0.2, 0.1])  

ranked_data, ranked_scores = fuzzy_topsis(data_continuous, weights, criteria)

print(f"Najlepszy punkt z danych: {ranked_data[0]} (Score: {ranked_scores[0]})")
print(f"Najgorszy punkt z danych: {ranked_data[-1]} (Score: {ranked_scores[-1]})")

# Example usage for discrete data
data_discrete = generate_elliptical_points_4d(center=(5, 10, 15, 20), axes=(2, 3, 4, 5), num_points=1000, angles=(30, 45))
objectives_data = [objectives(p) for p in data_discrete]
criteria = ['max', 'min', 'max', 'min']
weights = np.array([0.1, 0.3, 0.5, 0.1])

ranked_data, ranked_scores = fuzzy_topsis(objectives_data, weights, criteria)

print(f"Najlepszy punkt: {ranked_data[0]} (Score: {ranked_scores[0]:.4f})")
print(f"Najgorszy punkt: {ranked_data[-1]} (Score: {ranked_scores[-1]:.4f})")

# Visualization for continuous data
normalized_data = np.array(normalize_with_weights(data_continuous, weights, criteria))
pca = PCA(n_components=3)
reduced_data = pca.fit_transform([[f.m for f in row] for row in normalized_data])

# Najlepszy punkt po redukcji wymiarów
best_point_reduced = reduced_data[0]
colors = [row[0].u for row in normalized_data]

# Tworzenie wykresu 3D
scatter = go.Scatter3d(
    x=reduced_data[:, 0],
    y=reduced_data[:, 1],
    z=reduced_data[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, colorscale='Viridis', showscale=True),
    name='Punkty'
)

# Dodanie najlepszego punktu
best_point_marker = go.Scatter3d(
    x=[best_point_reduced[0]],
    y=[best_point_reduced[1]],
    z=[best_point_reduced[2]],
    mode='markers+text',
    marker=dict(size=10, color='red', symbol='diamond'),
    name='Najlepszy punkt',
    text=["Najlepszy punkt"],
    textposition='top center'
)

# Layout wykresu
layout = go.Layout(
    title="TOPSIS Dyskretny (4D, wizualizacja w 3D)",
    scene=dict(
        xaxis_title='Komponent 1',
        yaxis_title='Komponent 2',
        zaxis_title='Komponent 3'
    )
)

# Rysowanie wykresu
fig = go.Figure(data=[scatter, best_point_marker], layout=layout)
fig.show(renderer="browser")