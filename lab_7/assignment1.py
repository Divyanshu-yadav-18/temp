 def linear_regression(X, Y):
    mean_x = sum(X) / len(X)
    mean_y = sum(Y) / len(Y)

    # Slope (m) and intercept (c): y = mx + c
    m = sum((x - mean_x) * (y - mean_y) for x, y in zip(X, Y)) / sum(
        (x - mean_x) ** 2 for x in X
    )
    c = mean_y - m * mean_x
    return m, c


# Example usage
X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

m, c = linear_regression(X, Y)
predict = lambda x: m * x + c

print(f"Fit line: y = {m:.2f}x + {c:.2f}")
print(f"Prediction for x=6: {predict(6):.2f}")

A.2 K-Nearest Neighbors (KNN)
def knn_classify(X_train, y_train, x_query, k=3):
    # Compute squared Euclidean distance to all training samples
    distances = [
        (sum((a - b) ** 2 for a, b in zip(x, x_query)), label)
        for x, label in zip(X_train, y_train)
    ]
    # Sort and take k nearest labels
    k_nearest = [label for _, label in sorted(distances)[:k]]
    # Return majority class
    return max(set(k_nearest), key=k_nearest.count)


# Example usage
X_train = [[1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 6]]
y_train = ["A", "A", "A", "B", "B", "B"]

query = [2, 2]
print(f"KNN Prediction for {query}: {knn_classify(X_train, y_train, query, k=3)}")

A.3 Probabilistic Neural Network (PNN)
import math


def pnn_classify(X_train, y_train, x_query, sigma=1.0):
    classes = set(y_train)
    scores = {}

    for c in classes:
        # Filter samples belonging to class c
        c_samples = [x for x, label in zip(X_train, y_train) if label == c]

        # Parzen window / Gaussian kernel density sum
        kernel_sum = sum(
            math.exp(
                -sum((a - b) ** 2 for a, b in zip(x, x_query)) / (2 * sigma**2)
            )
            for x in c_samples
        )
        scores[c] = kernel_sum / len(c_samples)

    # Return class with the highest probability density score
    return max(scores, key=scores.get)


# Example usage
X_train = [[1, 1], [1.5, 2], [5, 6], [6, 5]]
y_train = [0, 0, 1, 1]

query = [1.2, 1.8]
print(
    f"PNN Prediction for {query}: {pnn_classify(X_train, y_train, query, sigma=0.5)}"
)
