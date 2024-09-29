import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# Define the Lorenz system
def lorenz(t, state, a=10, b=28, c=8/3):
    x, y, z = state
    dxdt = a * (y - x)
    dydt = x * (b - z) - y
    dzdt = x * y - c * z
    return [dxdt, dydt, dzdt]

# Set delta t
delta_t = 0.05
t_span = (0, 20)  # Time span
t_eval = np.arange(t_span[0], t_span[1], delta_t)  # Time evaluation with step size Delta t = 0.05
initial_state = [1, 1, 1]  # Initial conditions

# Solve the Lorenz system
solution = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)

# Extract x, y, z from the solution
x = solution.y[0]
y = solution.y[1]
z = solution.y[2]

# Normalize the data
x_normalized = (x - np.mean(x)) / np.std(x)
y_normalized = (y - np.mean(y)) / np.std(y)
z_normalized = (z - np.mean(z)) / np.std(z)

# Split data into training and testing sets (70% training, 30% testing)
split_index = int(0.7 * len(t_eval))

# Training data
t_train = t_eval[:split_index]
x_train = x_normalized[:split_index]
y_train = y_normalized[:split_index]
z_train = z_normalized[:split_index]

# Testing data
t_test = t_eval[split_index:]
x_test = x_normalized[split_index:]
y_test = y_normalized[split_index:]
z_test = z_normalized[split_index:]

# Save training and testing data as CSV
def save_data_as_csv(t, x, y, z, save_path, filename):
    data = pd.DataFrame({
        't': t,
        'x': x,
        'y': y,
        'z': z
    })
    data.to_csv(f'{save_path}/{filename}.csv', index=False)

# Save the training and testing data
save_path = r'D:/USC_Course/CSCE 790 Section 007 Neural Networks and Their Applications/chaotic dynamical systems/lorenz'
save_data_as_csv(t_train, x_train, y_train, z_train, save_path, 'lorenz_train_data')
save_data_as_csv(t_test, x_test, y_test, z_test, save_path, 'lorenz_test_data')

# Plot the results with the updated plot style (solid for actual, dashed for predicted)
plt.figure(figsize=(10, 6))

# Plot input x(t)
plt.subplot(3, 1, 1)
plt.plot(t_eval, x_normalized, label='Actual $\~x(t)$', color='blue', linestyle='-')
plt.ylabel('Input $\~x(t)$')
plt.legend()

# Plot output y(t)
plt.subplot(3, 1, 2)
plt.plot(t_eval, y_normalized, label='Actual $\~y(t)$', color='blue', linestyle='-')
plt.plot(t_eval, y_normalized, label='Reservoir output', color='red', linestyle='--')
plt.ylabel('Output and $\~y(t)$')
plt.legend()

# Plot output z(t)
plt.subplot(3, 1, 3)
plt.plot(t_eval, z_normalized, label='Actual $\~z(t)$', color='blue', linestyle='-')
plt.plot(t_eval, z_normalized, label='Reservoir output', color='red', linestyle='--')
plt.ylabel('Output and $\~z(t)$')
plt.xlabel('$t - T$')
plt.legend()

# Save plot as PDF
plt.tight_layout()
plt.savefig(f'{save_path}/lorenz_simulation_plot.pdf')
plt.show()
