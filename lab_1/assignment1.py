

players = {
    'Player A': [25, 30, 15, 20, 30, 45, 50, 35, 40, 30],
    'Player B': [50, 60, 55, 70, 65, 80, 75, 60, 90, 85],
    'Player C': [10, 12, 15, 10, 18, 20, 22, 15, 25, 30]
}

def calculateLength(arr):
    count = 0
    for _ in arr:
        count += 1
    return count

def customSum(arr):
    total = 0
    for val in arr:
        total += val
    return total

def customSort(arr):
    arr_copy = [x for x in arr]
    n = calculateLength(arr_copy)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr_copy[j] > arr_copy[j + 1]:
                arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
    return arr_copy

def calculateMean(arr):
    return customSum(arr) / calculateLength(arr)

def calculateMedian(arr):
    sorted_arr = customSort(arr)
    n = calculateLength(sorted_arr)
    if n % 2 != 0:
        return sorted_arr[n // 2]
    else:
        return (sorted_arr[(n // 2) - 1] + sorted_arr[n // 2]) / 2.0

def calculateMode(arr):
    counts = {}
    for val in arr:
        counts[val] = counts.get(val, 0) + 1
    max_freq = max(counts.values())
    return [val for val, freq in counts.items() if freq == max_freq]

def calc_variance(arr):
    mean = calculateMean(arr)
    n = calculateLength(arr)
    return customSum([(x - mean) ** 2 for x in arr]) / n

def calc_std_dev(arr):
    return calc_variance(arr) ** 0.5

def calc_z_scores(arr):
    mean = calculateMean(arr)
    std = calc_std_dev(arr)
    return [(x - mean) / std for x in arr] if std != 0 else [0.0 for _ in arr]

def run_assignment_1():
    print("=== ASSIGNMENT 1 RESULTS ===")
    for name, data in players.items():
        print(f"\n--- {name} ---")
        print(f"Mean:               {calculateMean(data):.2f}")
        print(f"Median:             {calculateMedian(data):.2f}")
        print(f"Mode:               {calculateMode(data)}")
        print(f"Variance:           {calc_variance(data):.2f}")
        print(f"Standard Deviation: {calc_std_dev(data):.2f}")

if __name__ == '__main__':
    run_assignment_1()
