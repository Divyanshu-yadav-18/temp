# assignment3.py
import matplotlib.pyplot as plt
from assignment1 import players, customSort, calculateMedian
from assignment2 import calc_quartiles_and_range

def run_assignment_3():
    print("\n=== ASSIGNMENT 3 RESULTS ===")
    
    # 1. Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, (name, data) in enumerate(players.items()):
        axes[idx].hist(data, bins=5, color='skyblue', edgecolor='black')
        axes[idx].set_title(f"Histogram - {name}")
    plt.tight_layout()
    plt.show()

    # 2. Quartile Diagram
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (name, data) in enumerate(players.items()):
        sorted_data = customSort(data)
        med_v = calculateMedian(data)
        _, q1, q3, _ = calc_quartiles_and_range(data)
        y = idx + 1
        
        ax.plot([sorted_data[0], sorted_data[-1]], [y, y], color='gray')
        ax.scatter([sorted_data[0], q1, med_v, q3, sorted_data[-1]], [y]*5, s=80)
        
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(players.keys())
    ax.set_title("Quartile Summary Diagram")
    plt.show()

if __name__ == '__main__':
    run_assignment_3()
