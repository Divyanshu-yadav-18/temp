import struct

# matplotlib is optional: if it is missing (no-library test), skip the plots
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

IMAGE_FILE = "train-images.idx3-ubyte"
COMPONENTS = 14      # number of principal components to keep (28 -> 14)
IMAGE_INDEX = 0      # which MNIST image to use


# ---------- Read one MNIST image ----------
def read_one_image(filename, index):
    with open(filename, "rb") as f:
        # header: magic number, image count, rows, cols (big-endian 32-bit ints)
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))

        # skip the header (16 bytes) and all earlier images
        f.seek(16 + index * rows * cols)

        image = []
        for i in range(rows):
            row = []
            for j in range(cols):
                pixel = f.read(1)[0] / 255.0     # scale 0-255 to 0-1
                row.append(pixel)
            image.append(row)
    return image


# ---------- Step 1: mean of each column ----------
def column_mean(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    mean = [0.0] * cols
    for j in range(cols):
        s = 0.0
        for i in range(rows):
            s += matrix[i][j]
        mean[j] = s / rows
    return mean


# ---------- Step 2: center the data (subtract column mean) ----------
def center(matrix, mean):
    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            row.append(matrix[i][j] - mean[j])
        result.append(row)
    return result


# ---------- Step 3: covariance matrix (cols x cols) ----------
# C[i][j] = sum over samples of (x_i * x_j) / (n - 1), data already centered
def covariance(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    cov = []
    for i in range(cols):
        cov_row = []
        for j in range(cols):
            s = 0.0
            for k in range(rows):
                s += matrix[k][i] * matrix[k][j]
            cov_row.append(s / (rows - 1))
        cov.append(cov_row)
    return cov


# ---------- Helper: dot product ----------
def dot(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


# ---------- Helper: matrix x vector ----------
def mat_vec(matrix, vector):
    return [dot(row, vector) for row in matrix]


# ---------- Helper: scale a vector to length 1 ----------
def normalize(v):
    length = dot(v, v) ** 0.5
    if length == 0:                 # avoid division by zero
        return v
    return [x / length for x in v]


# ---------- Step 4: eigenvector by power iteration ----------
# Repeating v = C*v and normalizing converges to the eigenvector
# with the largest eigenvalue. To get the NEXT eigenvector, we remove
# the part of v along earlier eigenvectors (Gram-Schmidt) every iteration.
def power_iteration(matrix, previous, iterations=300):
    n = len(matrix)
    # varied start vector (an all-ones start can be orthogonal to the answer)
    v = [1.0 + 0.01 * d for d in range(n)]

    for _ in range(iterations):
        # remove components along already-found eigenvectors
        for p in previous:
            proj = dot(v, p)
            for d in range(n):
                v[d] -= proj * p[d]

        v = mat_vec(matrix, v)      # multiply by covariance matrix
        v = normalize(v)            # keep length 1

    # eigenvalue = v^T C v (variance along direction v)
    eigenvalue = dot(v, mat_vec(matrix, v))
    return v, eigenvalue


# ---------- Find top-k principal components ----------
def pca_basis(cov, k):
    basis = []       # eigenvectors (principal directions)
    values = []      # eigenvalues (variance along each direction)
    for _ in range(k):
        vec, val = power_iteration(cov, basis)
        basis.append(vec)
        values.append(val)
    return basis, values


# ---------- Step 5: project data onto the basis (reduce dimension) ----------
def project(centered, basis):
    reduced = []
    for row in centered:
        coeff = [dot(row, b) for b in basis]    # one value per component
        reduced.append(coeff)
    return reduced


# ---------- Step 6: reconstruct approximate data from reduced form ----------
# x_hat = mean + sum of (coefficient * basis vector)
def reconstruct(reduced, basis, mean):
    reconstructed = []
    for i in range(len(reduced)):
        row = []
        for j in range(len(mean)):
            value = mean[j]
            for k in range(len(basis)):
                value += reduced[i][k] * basis[k][j]
            row.append(value)
        reconstructed.append(row)
    return reconstructed


# ======================= main =======================
image = read_one_image(IMAGE_FILE, IMAGE_INDEX)

mean = column_mean(image)
centered = center(image, mean)
cov = covariance(centered)

basis, eigenvalues = pca_basis(cov, COMPONENTS)

reduced = project(centered, basis)
reconstructed = reconstruct(reduced, basis, mean)

# ---------- Terminal output ----------
print("MNIST PCA IMPLEMENTATION")
print("Original Image Size : 28 x 28")
print("Reduced Dimensions  : 28 x", COMPONENTS)

print("\nTop 5 Eigenvalues")
for i in range(5):
    print("PC", i + 1, ":", round(eigenvalues[i], 4))

# total variance = trace of covariance matrix (sum of diagonal)
total_variance = sum(cov[i][i] for i in range(len(cov)))
retained = sum(eigenvalues) / total_variance * 100
print("\nVariance retained by", COMPONENTS, "components:", round(retained, 2), "%")

# mean squared reconstruction error
err = 0.0
for i in range(len(image)):
    for j in range(len(image[0])):
        err += (image[i][j] - reconstructed[i][j]) ** 2
print("Reconstruction MSE :", round(err / (len(image) * len(image[0])), 6))

# ---------- Display images (only if matplotlib exists) ----------
if HAS_PLT:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original 28x28")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(reduced, cmap="gray")
    plt.title("Reduced 28x" + str(COMPONENTS))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed, cmap="gray")
    plt.title("Reconstructed 28x28")
    plt.axis("off")

    plt.subplots_adjust(wspace=0.5)
    plt.show()
