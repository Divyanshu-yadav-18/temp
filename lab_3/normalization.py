from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data.tolist()

def normalize(X):
    rows = len(X)
    columns = len(X[0])

    minimum = []
    maximum = []

    for j in range(columns):
        low = X[0][j]
        high = X[0][j]
        for i in range(rows):
            if X[i][j] < low:
                low = X[i][j]

            if X[i][j] > high:
                high = X[i][j]

        minimum.append(low)
        maximum.append(high)

    normalized = []
    for i in range(rows):
        new_row = []
        for j in range(columns):
            diffence = maximum[j] - minimum[j]
            if diffence != 0:
                value = ((X[i][j] - minimum[j]) / diffence)
            else:
                value = 0.0
            new_row.append(value)
        normalized.append(new_row)
    return normalized

X_normalized = normalize(X)

print("Original First Sample:", X[0])
print("Normalized First Sample:", X_normalized[0])
print("Total Samples:", len(X_normalized))
print("Total Features:", len(X_normalized[0]))
