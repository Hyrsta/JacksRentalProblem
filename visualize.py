import numpy as np
import matplotlib.pyplot as plt

# Assuming 'policy' is your final policy matrix
# Adjust 'policy' accordingly based on your data
# Example policy matrix:
policy = np.array([[0, 0, -1, -1, -2, -2, -3, -3, -3, -4, -4, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5],
                   [1, 0, 0, -1, -1, -2, -2, -2, -3, -3, -4, -4, -5, -5, -5, -5, -5, -5, -5, -5, -5],
                   [1, 1, 0, 0, -1, -1, -1, -2, -2, -3, -3, -4, -4, -4, -5, -5, -5, -5, -5, -5, -5],
                   [2, 1, 1, 0, 0, 0, -1, -1, -2, -2, -3, -3, -3, -4, -4, -4, -4, -5, -5, -5, -5],
                   [2, 2, 1, 1, 0, 0, 0, -1, -1, -2, -2, -2, -3, -3, -3, -3, -4, -4, -4, -4, -4],
                   [3, 2, 2, 1, 1, 0, 0, 0, -1, -1, -1, -2, -2, -2, -2, -3, -3, -3, -3, -3, -3],
                   [3, 3, 2, 2, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -3],
                   [4, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -2, -2],
                   [4, 4, 3, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1],
                   [5, 4, 4, 4, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                   [5, 5, 5, 4, 4, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [5, 5, 5, 5, 4, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [5, 5, 5, 5, 4, 4, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [5, 5, 5, 5, 5, 4, 4, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [5, 5, 5, 5, 5, 5, 4, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [5, 5, 5, 5, 5, 5, 4, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [5, 5, 5, 5, 5, 5, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                   [5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 3, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0],
                   [5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 1, 1, 0, 0],
                   [5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 1, 1, 0]])

# Create a meshgrid for the state space with switched axes
cars_second_loc_vals = np.arange(0, 21, 1)
cars_first_loc_vals = np.arange(0, 21, 1)
cars_second_loc_mesh, cars_first_loc_mesh = np.meshgrid(cars_second_loc_vals, cars_first_loc_vals)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surface = ax.plot_surface(cars_first_loc_mesh, cars_second_loc_mesh, policy, cmap='viridis')

# Set ticks for X and Y axes with an interval of 4
ax.set_xticks(np.arange(0, 21, 4))
ax.set_yticks(np.arange(0, 21, 4))

# Set custom ticks for Z axis at intervals of 2, including 0, 5, and -5
ax.set_zticks(np.arange(-5, 6, 2))

# Add labels and title
ax.set_xlabel('Cars in First Location')
ax.set_ylabel('Cars in Second Location')
ax.set_zlabel('Action (Number of Cars Moved)')
ax.set_title('Optimal Policy')

# Add a colorbar
fig.colorbar(surface, ax=ax, shrink=0.6, aspect=10)

# Show the plot
plt.show()