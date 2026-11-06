age = [43, 21, 25, 42, 57, 59]
glucose = [99, 65, 79, 75, 87, 81]

n = len(age)

mean_age = sum(age) / n;
mean_glucose = sum(glucose) / n;

sum_prod = sum((age[i] - mean_age) * (glucose[i] - mean_glucose) for i in range(n))

sample_covariance = sum_prod / (n - 1)
population_covariance = sum_prod / n 

print(f"sample Covariance :{sample_covariance:.2f}")
print(f"population Covariance :{population_covariance:.2f}")
