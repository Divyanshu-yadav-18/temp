 
# ==========================================
# CHI-SQUARE GOODNESS OF FIT TEST
# Pure Python implementation
# No external libraries required
# ==========================================


# Step 1: Take input

n = int(input("Enter number of categories: "))

observed = []
expected = []

print("\nEnter frequencies for each category:")

for i in range(n):

    print(f"\nCategory {i + 1}")

    o = float(input("Observed frequency: "))
    e = float(input("Expected frequency: "))

    observed.append(o)
    expected.append(e)


# Step 2: Validate input

if n < 2:
    raise ValueError("At least 2 categories are required.")

if any(e <= 0 for e in expected):
    raise ValueError("Expected frequencies must be positive.")

if abs(sum(observed) - sum(expected)) > 1e-9:
    raise ValueError("Observed and expected totals must be equal.")


# Step 3: Calculate Chi-Square statistic

chi_square = 0.0

print("\n========== CALCULATIONS ==========")

for i in range(n):

    O = observed[i]
    E = expected[i]

    # Formula: (O - E)^2 / E
    contribution = ((O - E) ** 2) / E

    chi_square += contribution

    print(f"\nCategory {i + 1}")
    print(f"Observed  = {O}")
    print(f"Expected  = {E}")
    print(f"Contribution = {contribution:.4f}")


# Step 4: Calculate degrees of freedom

df = n - 1


# Step 5: Critical value lookup table
# Values are for alpha = 0.05 (right-tailed test).

critical_values = {
    1: 3.841,
    2: 5.991,
    3: 7.815,
    4: 9.488,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
    10: 18.307,
    11: 19.675,
    12: 21.026,
    13: 22.362,
    14: 23.685,
    15: 24.996,
    16: 26.296,
    17: 27.587,
    18: 28.869,
    19: 30.144,
    20: 31.410
}


# Step 6: Display results

print("\n========== RESULTS ==========")

print("Chi-Square Statistic:", round(chi_square, 4))
print("Degrees of Freedom:", df)
print("Significance Level: 0.05")

if df not in critical_values:

    print("Critical value unavailable for this df.")
    print("Extend the critical_values dictionary.")

else:

    critical = critical_values[df]

    print("Critical Value:", critical)

    # Step 7: Hypothesis decision

    if chi_square > critical:

        print("\nReject the Null Hypothesis (H0).")
        print("Observed frequencies differ significantly")
        print("from expected frequencies.")

    else:

        print("\nFail to Reject the Null Hypothesis (H0).")
        print("No statistically significant difference")
        print("was detected.")
  
