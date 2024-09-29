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

# Perceptron function
def perceptron(x1, x2, w1=-4.79, w2=5.90, b=-0.93):
    return w1 * x1 + w2 * x2 + b

# Generate grid of points
def generate_grid(num_points):
    x1 = np.linspace(-2, 2, num_points)
    x2 = np.linspace(-2, 2, num_points)
    X1, X2 = np.meshgrid(x1, x2)
    Z = perceptron(X1, X2)
    return X1, X2, Z

# Plotting function
def plot_surface(ax, X1, X2, Z, activation_func, title):
    surf = ax.plot_surface(X1, X2, activation_func(Z), cmap='viridis')
    ax.set_title(title, pad=20, fontsize=20)  # Increase the font size of the title to 20
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    return surf

# Function to generate and save the plots
def create_plots(num_points, file_name):
    X1, X2, Z = generate_grid(num_points)
    
    fig = plt.figure(figsize=(18, 6))  # 18 inches wide, 6 inches tall for one row of plots
    
    # Sigmoid plot
    ax1 = fig.add_subplot(131, projection='3d')  # 1st plot in the 1st column
    plot_surface(ax1, X1, X2, Z, sigmoid, 'Sigmoid')

    # Hard Limit plot
    ax2 = fig.add_subplot(132, projection='3d')  # 2nd plot in the 2nd column
    plot_surface(ax2, X1, X2, Z, hard_limit, 'Hard Limit')

    # RBF plot
    ax3 = fig.add_subplot(133, projection='3d')  # 3rd plot in the 3rd column
    plot_surface(ax3, X1, X2, Z, rbf, 'RBF')

    # Adjust layout to prevent overlap of labels and titles
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve some space at the top for titles

    # Save to a PDF file in the specified directory
    plt.savefig(file_name, format='pdf')

    # Show the plot on the screen
    plt.show()

# Set the directory path
directory_path = 'D:/USC_Course/CSCE 790 Section 007 Neural Networks and Their Applications/activation functions/Q_a/'

# Generate and save plots for 100, 5000, and 10000 sample points, and display them
create_plots(100, directory_path + 'activation_functions_100_points.pdf')
create_plots(5000, directory_path + 'activation_functions_5000_points.pdf')
create_plots(10000, directory_path + 'activation_functions_10000_points.pdf')
