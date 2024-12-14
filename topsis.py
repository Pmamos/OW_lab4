import numpy as np
import matplotlib.pyplot as plt


def objectives(u, p1=1, p2=1):
    x, y = u
    f1 = x**2 + y**2  
    f2 = (x - p1)**2 + (y + p2)**2  
    return np.array([f1, f2])

def topsis(decision_matrix, criteria):
    """
    Perform TOPSIS analysis on the given data.

    Parameters:
    - decision_matrix (np.ndarray): The decision matrix with alternatives as rows and criteria as columns.
    - criteria (np.ndarray): An array indicating if each criterion is beneficial (1) or non-beneficial (-1).

    Returns:
    - ranking (np.ndarray): The ranking of the alternatives.
    - scores (np.ndarray): The TOPSIS scores for each alternative.
    """
    # Normalize the decision matrix
    m, n = decision_matrix.shape
    divisors = np.empty(n)
    for j in range(n):
        column = decision_matrix[:, j]
        divisors[j] = np.linalg.norm(column, ord=np.inf)
    norm_matrix = decision_matrix / divisors

    # Determine the ideal best and worst solutions
    a_pos = np.zeros(n)
    a_neg = np.zeros(n)
    for j in range(n):
        column = norm_matrix[:, j]
        a_pos[j] = np.min(column) if criteria[j] == -1 else np.max(column)
        a_neg[j] = np.max(column) if criteria[j] == -1 else np.min(column)

    # Calculate distances to the ideal best and worst solutions
    cs = np.zeros(m)
    for i in range(m):
        diff_pos = norm_matrix[i] - a_pos
        diff_neg = norm_matrix[i] - a_neg
        sp = np.sqrt(np.sum(diff_pos ** 2))
        sn = np.sqrt(np.sum(diff_neg ** 2))
        cs[i] = sn / (sp + sn)

    # Rank the alternatives
    ranking = np.argsort(cs)[::-1]
    scores = cs

    return ranking, scores

# Example usage:
x_vals = np.linspace(-10, 10, 100)
y_vals = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x_vals, y_vals)
U_grid = np.column_stack([X.ravel(), Y.ravel()])
data = np.array([objectives(p, 1, 1) for p in U_grid])

criteria = np.array([1, -1])  # Example criteria (1 for beneficial, -1 for non-beneficial)

ranking, scores = topsis(data, criteria)

# Plot the results
best_point_index = ranking[0]
best_point = data[best_point_index]

plt.scatter(data[:, 0], data[:, 1], label="Wszystkie punkty")
plt.scatter(best_point[0], best_point[1], color='red', label="Najlepszy punkt", s=100)

plt.title("TOPSIS")
plt.xlabel("F1")
plt.ylabel("F2")
plt.legend()
plt.grid()
plt.show()

print(f"Ranking: {ranking}")
print(f"Scores: {scores}")
