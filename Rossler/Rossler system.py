import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.linear_model import Ridge
import pandas as pd

# Define the Rössler system
def rossler(state, t, a=0.5, b=2.0, c=4.0):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

# Generate training data using the Rössler system
def generate_rossler_data(T, dt):
    t = np.arange(0, T, dt)
    state0 = [1.0, 1.0, 1.0]
    states = odeint(rossler, state0, t)
    return t, states

# Define the reservoir computer
class ReservoirComputer:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=1.0, leakage_rate=1.0, regularization=1e-6):
        self.reservoir_size = reservoir_size
        self.leakage_rate = leakage_rate
        self.input_weights = np.random.uniform(-1.0, 1.0, (reservoir_size, input_size))
        self.reservoir_weights = np.random.uniform(-1.0, 1.0, (reservoir_size, reservoir_size))
        self.reservoir_weights *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.reservoir_weights)))
        self.output_weights = np.zeros((output_size, reservoir_size))
        self.regularization = regularization

    def train(self, inputs, outputs):
        reservoir_states = np.zeros((inputs.shape[0], self.reservoir_size))
        state = np.zeros(self.reservoir_size)
        for t in range(inputs.shape[0]):
            state = (1 - self.leakage_rate) * state + self.leakage_rate * np.tanh(
                np.dot(self.input_weights, inputs[t]) + np.dot(self.reservoir_weights, state)
            )
            reservoir_states[t] = state

        # Ridge regression to compute output weights
        ridge = Ridge(alpha=self.regularization, fit_intercept=False)
        ridge.fit(reservoir_states, outputs)
        self.output_weights = ridge.coef_

    def predict(self, inputs):
        reservoir_states = np.zeros((inputs.shape[0], self.reservoir_size))
        state = np.zeros(self.reservoir_size)
        predictions = []
        for t in range(inputs.shape[0]):
            state = (1 - self.leakage_rate) * state + self.leakage_rate * np.tanh(
                np.dot(self.input_weights, inputs[t]) + np.dot(self.reservoir_weights, state)
            )
            reservoir_states[t] = state
            predictions.append(np.dot(self.output_weights, state))
        return np.array(predictions)

# Save training and testing data as CSV
def save_data_to_csv(t_train, states_train, t_test, states_test, save_path):
    # Save training data
    train_data = pd.DataFrame({
        't': t_train,
        'x_train': states_train[:, 0],
        'y_train': states_train[:, 1],
        'z_train': states_train[:, 2]
    })
    train_data.to_csv(f'{save_path}/rossler_train_data.csv', index=False)

    # Save testing data
    test_data = pd.DataFrame({
        't': t_test,
        'x_test': states_test[:, 0],
        'y_test': states_test[:, 1],
        'z_test': states_test[:, 2]
    })
    test_data.to_csv(f'{save_path}/rossler_test_data.csv', index=False)

# Parameters
T_train = 260.0
T_test = 100.0
dt = 0.1
reservoir_size = 400
spectral_radius = 1.0
leakage_rate = 1.0
regularization = 1e-6

# Generate training and testing data
t_train, states_train = generate_rossler_data(T_train, dt)
t_test, states_test = generate_rossler_data(T_test, dt)

# Use the x-coordinate as input and y, z as the output to be predicted
input_train = states_train[:, 0].reshape(-1, 1)  # x-coordinate
output_train = states_train[:, 1:]  # y and z coordinates
input_test = states_test[:, 0].reshape(-1, 1)

# Initialize and train the reservoir computer
reservoir = ReservoirComputer(input_size=1, reservoir_size=reservoir_size, output_size=2, spectral_radius=spectral_radius, leakage_rate=leakage_rate, regularization=regularization)
reservoir.train(input_train, output_train)

# Make predictions
predictions = reservoir.predict(input_test)

# Save the training and testing data as CSV files
save_path = r'D:/USC_Course/CSCE 790 Section 007 Neural Networks and Their Applications/chaotic dynamical systems/'
save_data_to_csv(t_train, states_train, t_test, states_test, save_path)

# Plot the results and save the plot as a PDF
plt.figure(figsize=(12, 10))

# Plot input x(t)
plt.subplot(3, 1, 1)
plt.plot(t_test, input_test, 'b')
plt.title('Input: Preprocessed x(t)')
plt.xlabel('t')
plt.ylabel('x~(t)')

# Plot y-coordinate
plt.subplot(3, 1, 2)
plt.plot(t_test, states_test[:, 1], 'b', label='Actual y(t)')
plt.plot(t_test, predictions[:, 0], 'r--', label='Predicted y(t)')
plt.legend()
plt.xlabel('t')
plt.ylabel('y(t)')

# Plot z-coordinate
plt.subplot(3, 1, 3)
plt.plot(t_test, states_test[:, 2], 'b', label='Actual z(t)')
plt.plot(t_test, predictions[:, 1], 'r--', label='Predicted z(t)')
plt.legend()
plt.xlabel('t')
plt.ylabel('z(t)')

# Save the plot as a PDF
plt.tight_layout()
plt.savefig(f'{save_path}/rossler_simulation_plot.pdf')
plt.show()
