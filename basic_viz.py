import matplotlib.pyplot as plt
import numpy as np

""" Creates a basic 2D k-means visualization """

# Generate some random data to cluster
data = np.random.rand(100, 2)

# # Sample the initial centroids from the data points
# indices = np.random.choice(range(data.shape[0]), size=3, replace=False)
# centroids = data[indices]

# Initialize the centroids randomly
centroids = np.random.rand(3, 2)

# Initialize the plot
fig, ax = plt.subplots()

# Plot the data and the centroids
ax.scatter(data[:, 0], data[:, 1], label='Data Points')

plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
ax.set_axis_off()
plt.savefig('kmeans_before_clustering.png',dpi=300,bbox_inches='tight')

ax.scatter(centroids[:, 0], centroids[:, 1], label='Initial Centroids', marker='x', c=['b', 'g', 'r'])

# Save the initial plot
plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
ax.set_axis_off()
plt.savefig('kmeans_initial.png',dpi=300,bbox_inches='tight')

# Iterate over the algorithm
for i in range(1,15):
    # Clear the previous plot
    ax.clear()

    # Assign each data point to the nearest centroid
    distances = np.sqrt(np.sum((data - centroids[:, np.newaxis])**2, axis=2))
    clusters = np.argmin(distances, axis=0)
    
    # Plot the data points, coloring them by the cluster they belong to
    colors = ['b', 'g', 'r']
    ax.scatter(data[:, 0], data[:, 1], c=[colors[k] for k in clusters])
    
    # Plot the updated centroids, coloring them to match the data points in their corresponding cluster
    ax.scatter(centroids[:, 0], centroids[:, 1], label='Updated Centroids', marker='x', c=colors)
    
    ax.set_title(f'Iteration {i}: Assignment')

    # Save the updated plot
    # ax.set_axis_off()
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    ax.set_axis_off()
    plt.savefig(f'cluster_4/kmeans_{i}0.png',dpi=300,bbox_inches='tight')

    # Update the centroids to the mean of their assigned points
    for j in range(centroids.shape[0]):
        centroids[j] = data[clusters == j].mean(axis=0)

    # replot after cluster adjustment 
    ax.clear()
    # Plot the data points, coloring them by the cluster they belong to
    colors = ['b', 'g', 'r']
    ax.scatter(data[:, 0], data[:, 1], c=[colors[k] for k in clusters])

    # Plot the updated centroids, coloring them to match the data points in their corresponding cluster
    ax.scatter(centroids[:, 0], centroids[:, 1], label='Updated Centroids', marker='x', c=colors)
    
    ax.set_title(f'Iteration {i}: Update Centroid')

    # Save the updated plot
    # ax.set_axis_off()
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    ax.set_axis_off()
    plt.savefig(f'cluster_4/kmeans_{i}1.png',dpi=300,bbox_inches='tight')

plt.show()
