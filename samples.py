import numpy as np
import matplotlib.pyplot as plt

# Definindo o número de pontos
num_samples = 100

# Função para gerar pontos aleatórios dentro de um círculo com centro (center_x, center_y) e raio radius
def generate_random_points(center_x, center_y, radius, num_samples):
    angles = np.random.uniform(0, 2 * np.pi, num_samples)
    radii = np.sqrt(np.random.uniform(0, radius**2, num_samples))
    x = center_x + radii * np.cos(angles)
    y = center_y + radii * np.sin(angles)
    return np.column_stack((x, y))

# Gerando pontos aleatórios para as três nuvens de pontos
x1, y1 = generate_random_points(1, 1, 1, num_samples).T
x2, y2 = generate_random_points(3, -1, 1, num_samples).T
x3, y3 = generate_random_points(5, 1, 1, num_samples).T

# Criando as listas X e Y
X = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2)), np.column_stack((x3, y3))))
Y = np.hstack((np.full(num_samples, -1), np.full(num_samples, 0), np.full(num_samples, 1)))


# Embaralhando X e aplicando a mesma permutação em Y
shuffle_indices = np.random.permutation(len(X))
X = X[shuffle_indices]
Y = Y[shuffle_indices]

print("X embaralhado:", X)
print("Y embaralhado:", Y)



# Mapeando as classes para cores
colors = {-1: 'r', 0: 'g', 1: 'b'}
class_colors = [colors[y] for y in Y]


# Visualizando os pontos
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=class_colors)
plt.title("Três Nuvens de Pontos Aleatórios com Classes")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.xlim(-1,7)
plt.ylim(-5,5)
plt.show()

print("Shape do dataset (X):", X.shape)
print("Shape do dataset (Y):", Y.shape)