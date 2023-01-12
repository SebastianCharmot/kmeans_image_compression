# K-Means Algorithm for Image Compression

This Github repository contains code to run the k-means algorithm for image compression and create corresponding visualizations.

![](https://github.com/SebastianCharmot/kmeans_image_compression/blob/master/cuttlefish.gif)

## Getting Started

### Dependencies

All of the required libraries can be found in the requirements.txt file. 

### Understanding K-Means

The [basic_viz.py](https://github.com/SebastianCharmot/kmeans_image_compression/blob/master/basic_viz.py) file contains code that performs k-means in 2D space. It may be worthwhile to reference this code first if you are new to k-means. The [iterations_4.gif](https://github.com/SebastianCharmot/kmeans_image_compression/blob/master/iterations_4.gif) provides a visual explanation of how k-means works on a simple 2D example. 

<!-- ![](https://github.com/SebastianCharmot/kmeans_image_compression/blob/master/iterations_4.gif) -->

## K-Means for Image Compression

The main file to run k-means for image compression is [k_means.py](https://github.com/SebastianCharmot/kmeans_image_compression/blob/master/k_means.py). Simply modify the filepath in `parser.add_argument('--large_path', default='cuttlefish.jpg',
                        help='Path to large image')` to match your jpg file. The code in that file allows for users to create not only the compressed output image but also visualizations of the entire process, including the progression of the centroids. 

Below are some examples of the potential use cases:

### Visualizing the centroids and compressed images as k-means iterates

![](https://github.com/SebastianCharmot/kmeans_image_compression/blob/master/sea_turtle.gif)

### Comparing Different Values of $k$ 

![](https://github.com/SebastianCharmot/kmeans_image_compression/blob/master/comparing_k's.png)

### Visualizing the final centroids

![](https://github.com/SebastianCharmot/kmeans_image_compression/blob/master/comparing_centroids.png)



