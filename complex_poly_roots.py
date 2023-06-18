"""Finding roots of complex polynomials with numpy"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

degree = 10
image_res = 1024

# Define a complex polynomial
n = np.arange(degree)
rng = np.random.default_rng(0)
coeffs = rng.normal(size=degree) + 1j * rng.normal(size=degree)

# Generate a numpy Polynomial object to do the same thing
p = Polynomial(coeffs)
roots = p.roots()

# Define a complex domain to plot over
wiggle_room = 0.5  # Increase domain a bit beyond the furthest root by this factor
z_max = (1 + wiggle_room) * np.abs([roots.real, roots.imag]).max()
z_1d = np.linspace(-z_max, z_max, image_res)
z = z_1d[:, np.newaxis] + 1j * z_1d

# Evaluate the polynomial explicitly and using numpy
f_z = (coeffs * z[..., np.newaxis] ** n).sum(axis=-1)
p_z = p(z)


fig, axes = plt.subplots(ncols=3, nrows=2)
my_axes, np_axes = axes

# Plot my evaluation and numpy's to check they're the same
components = np.real, np.imag, lambda z: np.log(np.abs(z))
my_images = [
    ax.pcolormesh(z.real, z.imag, comp(f_z)) for ax, comp in zip(my_axes, components)
]
my_bars = [fig.colorbar(im, ax=ax) for im, ax in zip(my_images, my_axes)]

np_images = [
    ax.pcolormesh(z.real, z.imag, comp(p_z)) for ax, comp in zip(np_axes, components)
]
np_bars = [fig.colorbar(im, ax=ax) for im, ax in zip(np_images, np_axes)]

# Show roots on numpy evaluation
root_markers = [ax.scatter(roots.real, roots.imag, c="r", alpha=0.5) for ax in np_axes]

# Formatting and labelling
for ax in axes.flatten():
    ax.set(xlabel=r"$\Re(z)$", ylabel=r"$\Im(z)$", aspect="equal")

my_labels = r"$\Re(f(z))$", r"$\Im(f(z))$", r"$\ln|f(z)|$"
for bar, label in zip(my_bars, my_labels):
    bar.set_label(label)

np_labels = r"$\Re(p(z))$", r"$\Im(p(z))$", r"$\ln|p(z)|$"
for bar, label in zip(np_bars, np_labels):
    bar.set_label(label)

fig.tight_layout()
plt.show()
