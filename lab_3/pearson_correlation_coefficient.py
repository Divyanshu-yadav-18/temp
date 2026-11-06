import math 


age = [43, 21, 25, 42, 57, 59]
glucose = [99, 65, 79, 75, 87, 81]

def pearson_correlation(x,y):
    n = len(x)
    meanx = sum(x)/n
    meany = sum(y)/n

    numerator = sum((x[i] - meanx)*(y[i] - meany) for i in range(n))
    denom_x= sum((x[i] - meanx)**2 for i in range(n))
    denom_y = sum((y[i] - meany)**2 for i in range(n))

    r = numerator / math.sqrt(denom_x*denom_y)
    return r 

r_manual = pearson_correlation(age, glucose)
print(f"r: {r_manual: .4f}")

