# Raw Data
age = [43, 21, 25, 42, 57, 59]
glucose = [99, 65, 79, 75, 87, 81]

# Means
mean_age = sum(age) / len(age)
mean_glucose = sum(glucose) / len(glucose)

# Categorize into 2x2 observed frequencies
# matrix: [[low_age_low_g, low_age_high_g], [high_age_low_g, high_age_high_g]]
observed = [[0, 0], [0, 0]]

for a, g in zip(age, glucose):
    row = 0 if a < mean_age else 1
    col = 0 if g < mean_glucose else 1
    observed[row][col] += 1

row_totals = [sum(row) for row in observed]
col_totals = [sum(observed[r][c] for r in range(2)) for c in range(2)]
total = len(age)

# Compute Chi-Square
chi2 = 0.0
for r in range(2):
    for c in range(2):
        expected = (row_totals[r] * col_totals[c]) / total
        chi2 += ((observed[r][c] - expected) ** 2) / expected

print(f"Observed Table: {observed}")
print(f"Chi-Square Statistic: {chi2:.2f}")
