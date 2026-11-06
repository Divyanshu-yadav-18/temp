from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data.tolist()
names = iris.feature_names

# Mean
def mean(data):
    total = 0
    for x in data:
        total+=x
    return total/len(data)

# Median 
def median(data):
    arr = data.copy()
    n = len(arr)
    arr = sorted(arr)

    if n%2 == 1:
        return arr[n//2]
    return (arr[n // 2 - 1] + arr[n // 2])/2

# Mode
def mode(data):
    freq = {}

    for x in data:
        freq[x] = freq.get(x,0) + 1

    if not freq:
        return []
    
    max_count = max(freq.values())
    return [x for x, count in freq.items() if count == max_count]

print("Measures of Central Tendency")
for col in range(len(names)):
    data = [row[col] for row in X]

    print(names[col])
    print("Mean   :", round(mean(data), 3))
    print("Median :", round(median(data), 3))
    print("Mode   :", mode(data))
    print()
