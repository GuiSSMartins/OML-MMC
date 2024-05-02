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
    return x, y

# Gerando pontos aleatórios para as três nuvens de pontos
x1, y1 = generate_random_points(1, 1, 0.5, num_samples)
x2, y2 = generate_random_points(3, -1, 0.5, num_samples)
x3, y3 = generate_random_points(5, 1, 0.5, num_samples)

# Visualizando os pontos
plt.figure(figsize=(8, 6))
plt.scatter(x1, y1, label='Nuvem 1')
plt.scatter(x2, y2, label='Nuvem 2')
plt.scatter(x3, y3, label='Nuvem 3')
plt.title("Três Nuvens de Pontos Aleatórios")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Combinando as coordenadas x e y de todas as nuvens em um único array
data = np.vstack((np.column_stack((x1, y1)), np.column_stack((x2, y2)), np.column_stack((x3, y3))))
print("Shape do dataset:", data.shape)