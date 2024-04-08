import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Koch Snowflake
def koch_snowflake(n):
    if n == 0:
        return [[0, 0], [1, 0]]
    else:
        prev_lines = koch_snowflake(n-1)
        new_lines = []
        for p1, p2 in zip(prev_lines[:-1], prev_lines[1:]):
            p1 = np.array(p1)
            p2 = np.array(p2)
            new_lines.append(p1)
            new_lines.append(p1 + (p2 - p1) / 3)
            new_lines.append((p1 + p2) / 2 + np.array([-(p2 - p1)[1], (p2 - p1)[0]]) / 3)
            new_lines.append(p1 + 2 * (p2 - p1) / 3)
        new_lines.append(prev_lines[-1])
        return new_lines

# 2. Sierpinski Triangle
def sierpinski(n):
    if n == 0:
        return [[0, 0], [1, 0], [0.5, np.sqrt(3)/2]]
    else:
        prev_tri = sierpinski(n-1)
        new_tri = []
        for vertex in prev_tri:
            new_tri.append(vertex)
            new_tri.append([vertex[0] + 1, vertex[1]])
            new_tri.append([vertex[0] + 0.5, vertex[1] + np.sqrt(3)/2])
        return new_tri



# 4. Barnsley Fern (Barnsley's Fern)
def barnsley_fern(n):
    x, y = 0, 0
    points = [[x, y]]
    for _ in range(n):
        r = np.random.rand()
        if r <= 0.01:
            x, y = 0, 0.16 * y
        elif r <= 0.86:
            x, y = 0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6
        elif r <= 0.93:
            x, y = 0.20 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.6
        else:
            x, y = -0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44
        points.append([x, y])
    return points

# 5. Lorenz Attractor
def lorenz_attractor(sigma=10, rho=28, beta=8/3, dt=0.01, num_steps=10000):
    x, y, z = 0, 1, 1.05
    points = [[x, y, z]]
    for _ in range(num_steps):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x, y, z = x + dx, y + dy, z + dz
        points.append([x, y, z])
    return points

# 6. Menger Sponge (3D fractal)
def menger_sponge(n):
    if n == 0:
        return [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    else:
        prev_sponge = menger_sponge(n-1)
        new_sponge = []
        for p1, p2 in zip(prev_sponge[:-1], prev_sponge[1:]):
            new_sponge.extend([p1, p2])
            if np.linalg.norm(np.array(p1) - np.array(p2)) > 1:
                for i in range(1, 3):
                    new_sponge.append([(p1[0] * i + p2[0] * (3 - i)) / 3,
                                       (p1[1] * i + p2[1] * (3 - i)) / 3,
                                       (p1[2] * i + p2[2] * (3 - i)) / 3])
        return new_sponge



# 9. Mandelbrot Set
def mandelbrot(c, max_iter=100):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

def mandelbrot_fractal(width, height, zoom=1, x_off=0, y_off=0, max_iter=100):
    img = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            zx = 1.5 * (x - width / 2) / (0.5 * zoom * width) + x_off
            zy = (y - height / 2) / (0.5 * zoom * height) + y_off
            c = complex(zx, zy)
            img[x, y] = mandelbrot(c, max_iter)
    return img

# 10. Mandelbulb
def mandelbulb(x, y, z, max_iter=100):
    c = complex(x, y)
    z = complex(0, z)
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z**8 + c
        n += 1
    return n

def mandelbulb_fractal(size, max_iter=100):
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    z = np.linspace(-2, 2, size)
    fractal = np.zeros((size, size, size))
    for i in range(size):
        for j in range(size):
            for k in range(size):
                fractal[i, j, k] = mandelbulb(x[i], y[j], z[k], max_iter)
    return fractal

# Generate and plot each fractal
plt.figure(figsize=(15, 20))

# 1. Koch Snowflake
plt.subplot(5, 2, 1)
koch_points = koch_snowflake(4)
koch_xs, koch_ys = zip(*koch_points)
plt.plot(koch_xs, koch_ys, 'k')
plt.fill(koch_xs, koch_ys, 'k', alpha=0.3)
plt.title("Koch Snowflake")

# 2. Sierpinski Triangle
plt.subplot(5, 2, 2)
sierpinski_points = sierpinski(4)
sierpinski_xs, sierpinski_ys = zip(*sierpinski_points)
plt.plot(sierpinski_xs, sierpinski_ys, 'k')
plt.fill(sierpinski_xs, sierpinski_ys, 'k', alpha=0.3)
plt.title("Sierpinski Triangle")


# 4. Barnsley Fern
plt.subplot(5, 2, 3)
barnsley_points = barnsley_fern(10000)
barnsley_xs, barnsley_ys = zip(*barnsley_points)
plt.scatter(barnsley_xs, barnsley_ys, s=0.1, color='green')
plt.title("Barnsley Fern")

# 5. Lorenz Attractor
plt.subplot(5, 2, 4, projection='3d')
lorenz_points = np.array(lorenz_attractor())
plt.plot(lorenz_points[:,0], lorenz_points[:,1], lorenz_points[:,2], color='blue')
plt.title("Lorenz Attractor")

# 6. Menger Sponge (3D fractal)
ax = plt.subplot(5, 2, 5, projection='3d')
menger_points = np.array(menger_sponge(3))
ax.scatter(menger_points[:,0], menger_points[:,1], menger_points[:,2], color='orange')
ax.set_title("Menger Sponge")


# 9. Mandelbrot Set
plt.subplot(5, 2, 6)
mandelbrot_img = mandelbrot_fractal(300, 300, zoom=1, max_iter=100)
plt.imshow(mandelbrot_img.T, cmap='inferno', extent=[-2, 1, -1.5, 1.5])
plt.title("Mandelbrot Set")

# 10. Mandelbulb
plt.subplot(5, 2, 7)
mandelbulb_img = mandelbulb_fractal(20, max_iter=100)
plt.imshow(np.max(mandelbulb_img, axis=2), cmap='inferno', extent=[-2, 2, -2, 2])
plt.title("Mandelbulb")

plt.tight_layout()
plt.show()
