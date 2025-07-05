import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ================================================================
# Step 1: Define utility functions for wave generation
# ================================================================

def wavelength_to_freq(wavelength_nm):
    """Convert a wavelength in nanometers to spatial frequency in radians/pixel."""
    return 2 * np.pi / wavelength_nm

def generate_superposed_wave_image(lambda1_nm, lambda2_nm, size=64, noise_level=0.1):
    """
    Generate a synthetic image simulating polaritonic interference patterns
    by superposing two sinusoidal waves with added noise.

    Args:
        lambda1_nm (float): Wavelength of the first polariton mode (nm)
        lambda2_nm (float): Wavelength of the second polariton mode (nm)
        size (int): Size of the square image
        noise_level (float): Standard deviation of Gaussian noise

    Returns:
        2D numpy array representing the filtered superposed wave pattern
    """
    freq1 = wavelength_to_freq(lambda1_nm)
    freq2 = wavelength_to_freq(lambda2_nm)
    x = np.linspace(0, 2 * np.pi, size)
    X, Y = np.meshgrid(x, x)
    wave1 = np.sin(freq1 * X)
    wave2 = np.sin(freq2 * X + np.pi / 4)  # phase shift adds variety
    image = wave1 + wave2 + noise_level * np.random.randn(*X.shape)
    return gaussian_filter(image, sigma=1)  # simulate instrument smoothing

# ================================================================
# Step 2: Create synthetic dataset with labeled wavelength pairs
# ================================================================

num_samples = 1000
image_height, image_width = 64, 64
X = np.zeros((num_samples, image_height, image_width, 1))  # input images
y = np.zeros((num_samples, 2))  # target labels: [lambda1, lambda2] in nm

for i in range(num_samples):
    lambda1 = np.random.uniform(300, 600)  # longer wavelength range (e.g., Mode 1)
    lambda2 = np.random.uniform(200, 400)  # shorter wavelength range (e.g., Mode 2)
    image = generate_superposed_wave_image(lambda1, lambda2, size=image_height)
    X[i, :, :, 0] = image
    y[i] = [lambda1, lambda2]  # true labels

# ================================================================
# Step 3: Define and compile a CNN for regression
# ================================================================

model = models.Sequential([
    layers.Input(shape=(image_height, image_width, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)  # Output: predicted [lambda1, lambda2]
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the synthetic dataset
model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

# ================================================================
# Step 4: Test the trained model on new synthetic samples
# ================================================================

X_new = np.zeros((10, image_height, image_width, 1))
y_true = np.zeros((10, 2))

for i in range(10):
    lambda1 = np.random.uniform(300, 600)
    lambda2 = np.random.uniform(200, 400)
    image = generate_superposed_wave_image(lambda1, lambda2, size=image_height)
    X_new[i, :, :, 0] = image
    y_true[i] = [lambda1, lambda2]

# Predict wavelengths from new images
y_pred = model.predict(X_new)

# ================================================================
# Step 5: Visualize predictions versus ground truth (images)
# ================================================================

fig, axs = plt.subplots(2, 5, figsize=(20, 6))
axs = axs.flatten()

for i in range(10):
    axs[i].imshow(X_new[i, :, :, 0], cmap='inferno')
    axs[i].axis('off')
    axs[i].set_title(
        f"Pred: {y_pred[i][0]:.1f}, {y_pred[i][1]:.1f}\nTrue: {y_true[i][0]:.1f}, {y_true[i][1]:.1f}"
    )

plt.tight_layout()
plt.show()

# ================================================================
# Step 6: Plot true and predicted wavelength pairs with fit
# ================================================================

def fit_func(x, a, b):
    return a * x**0.5 + b

x_true = y_true[:, 0]  # lambda1
y_true_vals = y_true[:, 1]  # lambda2
x_pred = y_pred[:, 0]
y_pred_vals = y_pred[:, 1]

# Fit a curve to the true values
popt, _ = curve_fit(fit_func, x_true, y_true_vals)
x_fit = np.linspace(min(x_true), max(x_true), 100)
y_fit = fit_func(x_fit, *popt)

# Linear regression fit
x_true_reshaped = x_true.reshape(-1, 1)
lin_reg = LinearRegression()
lin_reg.fit(x_true_reshaped, y_true_vals)
y_lin_fit = lin_reg.predict(x_fit.reshape(-1, 1))

# Compute metrics
r2 = r2_score(y_true_vals, lin_reg.predict(x_true_reshaped))
mse = mean_squared_error(y_true_vals, lin_reg.predict(x_true_reshaped))


# Plot
# Plot everything together
# Plot everything together
plt.figure(figsize=(8, 6))
plt.scatter(x_true, y_true_vals, label='True (circles)', color='blue', s=80, facecolors='none')
plt.plot(x_fit, y_fit, 'k--', label='Nonlinear fit (sqrt)')
plt.plot(x_fit, y_lin_fit, 'g-.', label='Linear fit')
plt.scatter(x_pred, y_pred_vals, label='Predicted (X)', color='red', marker='x', s=80)
plt.xlabel('λ₁ (nm)')
plt.ylabel('λ₂ (nm)')
plt.title('True vs Predicted Wavelength Pairs with Regression Fits')
plt.legend()
plt.grid(True)

# Show R² and MSE on the plot
plt.text(0.05, 0.95, f"Linear R² = {r2:.3f}\nLinear MSE = {mse:.1f}",
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
for ax in axs.flat:
    ax.tick_params(axis='both', labelsize=7)

plt.tight_layout()
plt.show()
