import matplotlib.pyplot as plt
from assignment1 import players, customSort, calculateLength

def calc_quartiles_and_range(arr):
    sorted_arr = customSort(arr)
    n = calculateLength(sorted_arr)
    
    r = sorted_arr[-1] - sorted_arr[0]
    
    pos1 = (n - 1) * 0.25
    idx1 = int(pos1)
    q1 = sorted_arr[idx1] + (pos1 - idx1) * (sorted_arr[idx1 + 1] - sorted_arr[idx1]) if idx1 + 1 < n else sorted_arr[idx1]
    
    pos3 = (n - 1) * 0.75
    idx3 = int(pos3)
    q3 = sorted_arr[idx3] + (pos3 - idx3) * (sorted_arr[idx3 + 1] - sorted_arr[idx3]) if idx3 + 1 < n else sorted_arr[idx3]
    
    return r, q1, q3, q3 - q1

def run_assignment_2():
    print("\n=== ASSIGNMENT 2 RESULTS ===")
    plot_data, labels = [], []
    
    for name, data in players.items():
        r, q1, q3, iqr = calc_quartiles_and_range(data)
        print(f"\n--- {name} ---")
        print(f"Range: {r:.2f} | Q1: {q1:.2f} | Q3: {q3:.2f} | IQR: {iqr:.2f}")
        plot_data.append(data)
        labels.append(name)
        
    plt.figure(figsize=(8, 6))
    plt.boxplot(plot_data, tick_labels=labels)
    plt.title("Box Plot of Players' Performance")
    plt.show()

if __name__ == '__main__':
    run_assignment_2()

