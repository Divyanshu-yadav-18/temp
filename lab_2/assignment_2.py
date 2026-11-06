import urllib.request
import math

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

print("Downloading Iris dataset from UCI repository...")
req = urllib.request.urlopen(url)
raw_data = req.read().decode('utf-8').strip().split('\n')

features = []
labels = []
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

for line in raw_data:
    if line.strip():
        parts = line.strip().split(',')
        # Convert first 4 columns to float
        features.append([float(x) for x in parts[:4]])
        labels.append(parts[4])

num_rows = len(features)
num_cols = len(features[0])

print(f"Dataset successfully loaded: {num_rows} samples, {num_cols} features.\n")


def get_column(matrix, col_idx):
    return [row[col_idx] for row in matrix]

def calc_mean(col):
    return sum(col) / len(col)

def calc_std(col, mean):
    variance = sum((x - mean) ** 2 for x in col) / len(col)
    return math.sqrt(variance)


def min_max_normalization(matrix):
    normalized = [[0.0] * num_cols for _ in range(num_rows)]
    
    for j in range(num_cols):
        col = get_column(matrix, j)
        min_val = min(col)
        max_val = max(col)
        denom = max_val - min_val
        
        for i in range(num_rows):
            normalized[i][j] = (matrix[i][j] - min_val) / denom if denom != 0 else 0.0
            
    return normalized


def z_score_normalization(matrix):
    normalized = [[0.0] * num_cols for _ in range(num_rows)]
    
    for j in range(num_cols):
        col = get_column(matrix, j)
        mean_val = calc_mean(col)
        std_val = calc_std(col, mean_val)
        
        for i in range(num_rows):
            normalized[i][j] = (matrix[i][j] - mean_val) / std_val if std_val != 0 else 0.0
            
    return normalized


def decimal_scaling_normalization(matrix):
    normalized = [[0.0] * num_cols for _ in range(num_rows)]
    
    for j in range(num_cols):
        col = get_column(matrix, j)
        max_abs = max(abs(x) for x in col)
        
        # Find j = smallest integer such that max(|v'|) < 1
        j_scale = math.ceil(math.log10(max_abs)) if max_abs > 0 else 1
        divisor = 10 ** j_scale
        
        for i in range(num_rows):
            normalized[i][j] = matrix[i][j] / divisor
            
    return normalized

min_max_data = min_max_normalization(features)
z_score_data = z_score_normalization(features)
decimal_data = decimal_scaling_normalization(features)

def print_head(title, data, n=5):
    print(f"=== {title} (First {n} rows) ===")
    header = "  ".join([f"{name:>12}" for name in feature_names])
    print(f"{'Index':<6} {header}")
    print("-" * 60)
    for i in range(n):
        row_str = "  ".join([f"{val:12.4f}" for val in data[i]])
        print(f"Row {i:<2} {row_str}")
    print("\n")

print_head("RAW DATA", features)
print_head("1. MIN-MAX NORMALIZATION [0, 1]", min_max_data)
print_head("2. Z-SCORE NORMALIZATION", z_score_data)
print_head("3. DECIMAL SCALING", decimal_data)
