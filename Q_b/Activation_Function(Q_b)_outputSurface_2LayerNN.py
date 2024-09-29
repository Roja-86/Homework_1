import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def hard_limit(z):
    return np.where(z >= 0, 1, 0)

def rbf(z):
    return np.exp(-z**2)

# Two-layer NN function based on the question
def two_layer_nn(x1, x2, V, W, bv, bw, activation_func):
    # Compute the inner product with V and add bias bv
    z = V[0, 0] * x1 + V[0, 1] * x2 + bv[0]
    z2 = V[1, 0] * x1 + V[1, 1] * x2 + bv[1]
    
    # Apply the activation function
    a = activation_func(np.array([z, z2]))
    
    # Compute the output with the outer layer weights W and bias bw
    y = W[0] * a[0] + W[1] * a[1] + bw
    return y

# Parameters given in the question
V = np.array([[-2.69, -2.80], [-3.39, -4.56]])
W = np.array([-4.91, 4.95])
bv = np.array([-2.21, 4.76])
bw = -2.28

# Generate grid of points
def generate_grid(num_points):
    x1 = np.linspace(-2, 2, num_points)
    x2 = np.linspace(-2, 2, num_points)
    X1, X2 = np.meshgrid(x1, x2)
    return X1, X2

# Plotting function
def plot_surface(ax, X1, X2, Z, title):
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis')
    ax.set_title(title, pad=20, fontsize=20)  # Increase the font size of the title to 20
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    return surf

# Function to generate and save the plots
def create_plots(num_points, file_name):
    X1, X2 = generate_grid(num_points)
    
    fig = plt.figure(figsize=(18, 6))  # 18 inches wide, 6 inches tall for one row of plots
    
    # Sigmoid plot
    Z_sigmoid = two_layer_nn(X1, X2, V, W, bv, bw, sigmoid)
    ax1 = fig.add_subplot(131, projection='3d')  # 1st plot in the 1st column
    plot_surface(ax1, X1, X2, Z_sigmoid, 'Sigmoid')

    # Hard Limit plot
    Z_hard_limit = two_layer_nn(X1, X2, V, W, bv, bw, hard_limit)
    ax2 = fig.add_subplot(132, projection='3d')  # 2nd plot in the 2nd column
    plot_surface(ax2, X1, X2, Z_hard_limit, 'Hard Limit')

    # RBF plot
    Z_rbf = two_layer_nn(X1, X2, V, W, bv, bw, rbf)
    ax3 = fig.add_subplot(133, projection='3d')  # 3rd plot in the 3rd column
    plot_surface(ax3, X1, X2, Z_rbf, 'RBF')

    # Adjust layout to prevent overlap of labels and titles
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve some space at the top for titles

    # Save to a PDF file in the specified directory
    plt.savefig(file_name, format='pdf')

    # Show the plot on the screen
    plt.show()

# Set the directory path
directory_path = 'D:/USC_Course/CSCE 790 Section 007 Neural Networks and Their Applications/activation functions/Q_b/'

# Generate and save plots for 100, 5000, and 10000 sample points, and display them
create_plots(100, directory_path + 'activation_functions_100_points.pdf')
create_plots(5000, directory_path + 'activation_functions_5000_points.pdf')
create_plots(10000, directory_path + 'activation_functions_10000_points.pdf')

