# import matplotlib.pyplot as plt
# import numpy as np

# # floorplan_fuse_map
# floorplan_fuse_map = {
#     1: [0, 0, 0],            # background
#     2: [192, 192, 224],      # closet
#     3: [192, 255, 255],      # batchroom/washroom
#     4: [224, 255, 192],      # livingroom/kitchen/dining room
#     5: [255, 224, 128],      # bedroom
#     6: [255, 160, 96],       # hall
#     7: [255, 224, 224],      # balcony
#     8: [255, 60, 128],       # extra label for opening (door&window)
#     9: [255, 255, 255]      # extra label for wall line
# }

# # 可视化颜色
# fig, ax = plt.subplots(1, len(floorplan_fuse_map), figsize=(15, 2))

# for i, (label, color) in enumerate(floorplan_fuse_map.items()):
#     ax[i].imshow(np.array([[color]], dtype=np.uint8))
#     ax[i].set_title(f"Label {label}")
#     ax[i].axis('off')

# plt.savefig('result.png')


import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(12399)

# Number of clusters and points per cluster
num_clusters = 6
points_per_cluster = 64

# Increase the figure size and ensure the clusters do not overlap
plt.figure(figsize=(12, 12))

# Create random cluster centers ensuring they are far apart
cluster_centers = np.random.rand(num_clusters, 2) * 100

# Generate random points around each cluster center
points = []
for center in cluster_centers:
    # Random points around the center with more spread
    cluster_points = center + np.random.randn(points_per_cluster, 2) * 1.2
    points.append(cluster_points)

# Flatten the list of points and assign cluster labels
points = np.vstack(points)
labels = np.hstack([[i] * points_per_cluster for i in range(num_clusters)])

# Plot the scatter plot with different colors and add more "texture" to the points
for i in range(num_clusters):
    plt.scatter(
        points[labels == i, 0], 
        points[labels == i, 1], 
        label=f'Cluster {i+1}', 
        alpha=0.8,  # Adjust the transparency
        edgecolors='k',  # Add a black edge to the points
        linewidths=0.5,  # Set the width of the edges
        s=100  # Increase the size of the points for better visibility
    )

# Remove the x and y axis tick numbers
plt.xticks([])
plt.yticks([])

plt.savefig('tttttt.png', dpi=300)  # Save the figure with high resolution
plt.show()



# import matplotlib.pyplot as plt
# import numpy as np

# # Define the number of categories and subcategories
# categories = 6
# subcategories_per_category = 4
# points_per_subcategory = 20

# # Colors for the categories
# colors = plt.cm.get_cmap('tab10', categories)

# # Markers for the subcategories
# markers = ['o', 's', 'D', '^']

# # Generate random data with clear separation between categories and subcategories
# np.random.seed(42)  # for reproducibility
# data = []
# labels = []

# for i in range(categories):
#     category_center = np.random.rand(2) * 20  # Increased the range to spread categories further apart
#     for j in range(subcategories_per_category):
#         subcategory_center = category_center + np.random.randn(2) * 0.5  # Slight offset for subcategories
#         points = subcategory_center + np.random.randn(points_per_subcategory, 2) * 0.1  # Tight cluster around subcategory center
#         data.append(points)
#         labels.extend([(i, j)] * points_per_subcategory)

# data = np.vstack(data)
# labels = np.array(labels)

# # Plotting the scatter plot
# plt.figure(figsize=(10, 8))

# for i in range(categories):
#     for j in range(subcategories_per_category):
#         idx = (labels[:, 0] == i) & (labels[:, 1] == j)
#         plt.scatter(data[idx, 0], data[idx, 1], color=colors(i), marker=markers[j], label=f'Category {i+1}, Subcategory {j+1}', edgecolor='black')

# plt.title('Scatter Plot with Six Categories and Four Subcategories Each')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.savefig('tttt.png')
