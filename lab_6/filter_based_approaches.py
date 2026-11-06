


# ---------- Helper: log base 2 without the math module ----------
# Reduce x to [1, 2), then use ln(x) = 2 * (z + z^3/3 + z^5/5 + ...), z = (x-1)/(x+1)
def log2(x):
    k = 0
    while x < 1:                    # scale up until x >= 1
        x *= 2
        k -= 1
    while x >= 2:                   # scale down until x < 2
        x /= 2
        k += 1
    z = (x - 1) / (x + 1)
    s = 0.0
    term = z
    for i in range(1, 60, 2):       # odd powers: z, z^3, z^5, ...
        s += term / i
        term *= z * z
    ln_x = 2 * s
    return k + ln_x / 0.6931471805599453    # ln(x)/ln(2) + k


# ---------- Helper: column of a matrix ----------
def column(X, j):
    return [row[j] for row in X]


# ---------- Helper: put a value into one of 'bins' equal-width bins ----------
# Chi-square and information gain need categories, so continuous values are binned
def bin_index(val, mn, width, bins):
    if width == 0:                  # constant feature: everything in bin 0
        return 0
    return min(int((val - mn) / width), bins - 1)    # max value goes to the last bin


# ---------- FILTER 1: Variance ----------
# variance = average of (x - mean)^2
def variance(data):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / n


# ---------- FILTER 2: Pearson correlation ----------
# r = covariance(x, y) / (std_x * std_y), always between -1 and +1
def correlation(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den != 0 else 0.0


# ---------- FILTER 3: Chi-square ----------
# chi2 = sum of (O - E)^2 / E, E = row_total * col_total / n
def chi_square(feature, labels, bins):
    mn, mx = min(feature), max(feature)
    width = (mx - mn) / bins
    classes = sorted(set(labels))
    total = len(labels)

    # contingency table: rows = bins, columns = classes
    table = [[0] * len(classes) for _ in range(bins)]
    for val, lab in zip(feature, labels):
        b = bin_index(val, mn, width, bins)
        table[b][classes.index(lab)] += 1

    row_tot = [sum(r) for r in table]
    col_tot = [sum(table[b][c] for b in range(bins)) for c in range(len(classes))]

    chi = 0.0
    for b in range(bins):
        for c in range(len(classes)):
            E = row_tot[b] * col_tot[c] / total      # expected count if independent
            if E > 0:
                chi += (table[b][c] - E) ** 2 / E
    return chi


# ---------- FILTER 4: Information gain ----------
# entropy H = -sum p * log2(p)
def entropy(labels):
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1         # frequency dictionary
    total = len(labels)
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * log2(p)
    return h


# IG = H(y) - H(y | feature)
def information_gain(feature, labels, bins):
    mn, mx = min(feature), max(feature)
    width = (mx - mn) / bins
    total = len(labels)

    groups = [[] for _ in range(bins)]               # labels that fall in each bin
    for val, lab in zip(feature, labels):
        groups[bin_index(val, mn, width, bins)].append(lab)

    h_cond = 0.0                                     # weighted entropy after the split
    for g in groups:
        if g:
            h_cond += (len(g) / total) * entropy(g)
    return entropy(labels) - h_cond


# ======================= Input =======================
n = int(input("Number of samples: "))
d = int(input("Number of features: "))
names = input("Feature names (space separated, or press Enter for default): ").split()
if len(names) != d:
    names = ["f" + str(j) for j in range(d)]

X, y = [], []
for i in range(n):
    vals = input(f"Sample {i + 1} ({d} feature values then class label): ").split()
    X.append([float(v) for v in vals[:d]])
    y.append(vals[d])                                # label kept as text (any class name)

bins = int(input("Number of bins for chi-square / info gain (e.g. 3): "))
var_thr = float(input("Variance threshold (drop if below): "))
corr_thr = float(input("Correlation threshold (e.g. 0.9): "))
k = int(input("How many features to finally keep: "))

# ======================= Compute all scores =======================
var_scores = [variance(column(X, j)) for j in range(d)]
chi_scores = [chi_square(column(X, j), y, bins) for j in range(d)]
ig_scores = [information_gain(column(X, j), y, bins) for j in range(d)]

print("\n=== 1. VARIANCE ===")
for j in range(d):
    print(f"{names[j]}: {var_scores[j]:.4f}")

print("\n=== 2. CORRELATION (feature vs feature) ===")
corr = [[correlation(column(X, a), column(X, b)) for b in range(d)] for a in range(d)]
for a in range(d):
    for b in range(a + 1, d):
        print(f"{names[a]} & {names[b]}: {corr[a][b]:.4f}")

print("\n=== 3. CHI-SQUARE ===")
for j in range(d):
    print(f"{names[j]}: {chi_scores[j]:.4f}")

print("\n=== 4. INFORMATION GAIN ===")
for j in range(d):
    print(f"{names[j]}: {ig_scores[j]:.4f}")

# ======================= Combine the filters =======================
# Step A: variance filter - keep only features that vary enough
kept = [j for j in range(d) if var_scores[j] >= var_thr]
print("\nAfter variance filter:", [names[j] for j in kept])

# Step B: correlation filter - for each highly correlated pair, drop the one
# with the lower information gain (it is the less useful copy)
removed = set()
for a in kept:
    for b in kept:
        if a < b and a not in removed and b not in removed:
            if abs(corr[a][b]) > corr_thr:
                drop = a if ig_scores[a] < ig_scores[b] else b
                removed.add(drop)
                print(f"  {names[a]} & {names[b]} correlated -> dropping {names[drop]}")
kept = [j for j in kept if j not in removed]
print("After correlation filter:", [names[j] for j in kept])

# Step C: rank the remaining features by information gain, keep the top k
ranked = sorted(kept, key=lambda j: ig_scores[j], reverse=True)
final = ranked[:k]

print("\nRanking (information gain):")
for j in ranked:
    print(f"  {names[j]}: IG = {ig_scores[j]:.4f}, chi2 = {chi_scores[j]:.4f}")

print("\nSELECTED FEATURES:", [names[j] for j in final])
