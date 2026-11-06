import random

# ---------- Demo data: f0 informative, f1 near copy of f0, f2 constant, f3 noise ----------
X = [[1.0, 2.1, 5, 3.0], [1.2, 2.3, 5, 1.0], [0.9, 1.9, 5, 2.0], [1.1, 2.2, 5, 1.5],
     [5.0, 10.1, 5, 2.5], [5.2, 10.4, 5, 1.2], [4.9, 9.8, 5, 3.1], [5.1, 10.2, 5, 2.2],
     [5.3, 10.5, 5, 1.8]]
y = ["A", "A", "A", "A", "B", "B", "B", "B", "B"]


# ---------- Fitness: leave-one-out 1-NN accuracy on selected features ----------
def fitness(chromosome, X, y, penalty=0.02):
    selected = [i for i, bit in enumerate(chromosome) if bit == 1]
    if not selected:                          # empty subset is useless
        return 0.0

    correct = 0
    for i in range(len(X)):
        best_dist, best_label = None, None
        for j in range(len(X)):
            if i == j:                        # leave the sample itself out
                continue
            # squared distance using ONLY the selected features
            dist = sum((X[i][f] - X[j][f]) ** 2 for f in selected)
            if best_dist is None or dist < best_dist:
                best_dist, best_label = dist, y[j]
        if best_label == y[i]:
            correct += 1

    accuracy = correct / len(X)
    # small penalty so that, at equal accuracy, fewer features win
    return accuracy - penalty * len(selected) / len(chromosome)


def genetic_algorithm(X, y, pop_size=10, max_gen=30, mutation_rate=0.1, patience=8):
    n_features = len(X[0])

    # 1. random initial population of bit-strings
    population = [[random.randint(0, 1) for _ in range(n_features)]
                  for _ in range(pop_size)]

    best_chrom, best_fit = None, -1.0
    best_history, avg_history = [], []
    stall = 0                                  # generations without improvement

    for gen in range(max_gen):
        # 2. evaluate every chromosome ONCE
        fits = [fitness(ind, X, y) for ind in population]

        # track the best chromosome seen so far
        improved = False
        for ind, f in zip(population, fits):
            if f > best_fit + 1e-9:
                best_fit, best_chrom, improved = f, ind[:], True
        stall = 0 if improved else stall + 1

        best_history.append(best_fit)
        avg_history.append(sum(fits) / len(fits))
        print(f"Gen {gen:2d}: best = {best_fit:.3f}, avg = {avg_history[-1]:.3f}, "
              f"subset = {best_chrom}")

        # early stop: best fitness has not improved for 'patience' generations
        if stall >= patience:
            print(f"Stopped at generation {gen} (no improvement)")
            break

        # 3. build the next generation
        new_pop = [best_chrom[:]]              # elitism: keep the best unchanged
        while len(new_pop) < pop_size:
            # tournament selection: best of 3 random chromosomes (reuse stored fits)
            def pick():
                contenders = random.sample(range(pop_size), 3)
                return population[max(contenders, key=lambda c: fits[c])]

            p1, p2 = pick(), pick()

            # single-point crossover
            point = random.randint(1, n_features - 1)
            child = p1[:point] + p2[point:]

            # mutation: flip each bit with small probability
            for i in range(n_features):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            new_pop.append(child)
        population = new_pop

    return best_chrom, best_fit, best_history, avg_history


print("=== GA Feature Selection ===")
chrom, fit, best_hist, avg_hist = genetic_algorithm(X, y)

print("\nBest chromosome :", chrom)
print("Selected features:", [i for i, b in enumerate(chrom) if b == 1])
print("Best fitness     :", round(fit, 3))

