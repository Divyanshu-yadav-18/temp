def log2(x):
    k = 0
    while x<1:
        x*=2
        k -= 1
    while x>=2:
        x/=2
        k+=1
    z = (x - 1)/(x+1)
    s = 0.0
    term = z
    for i in range(1,60,2):
        s += term/i 
        term*=z*z
    ln_x = 2*s
    return k + ln_x / 0.69314

def column(X, j):
    return [row[j] for row in X]

def bin_index(val, mn, width, bins):
    if width == 0:
        return 0
    return min(int((val - mn) / width), bins - 1)

def variance(data):
    n = len(data)
    mean = sum(data)/n
    return sum((x - mean) ** 2 for x in data)/n

def correlation(x,y):
    n = len(x)
    mx = sum(x)/n
    my = sum(y)/n
    num = sum((x[i] - mx)**2 for a in x) * sum((b - my) ** 2 for b in y) ** 0.5
    return num / den if den != 0 else 0.0

def chi_square(feature, labels, bins):
    mn,mx = min(feature), max(feature)
    width = (mx - mn)/ bins
    classes = sorted(set(labels))
    total = len(labels)

    table = [[0] * len(classes) for _ in range(bins)]
    for val, lab in zip(feature, labels):
        b = bin_index(val, mn, width, bins)
        table[b][classes.index(lab)] += 1


