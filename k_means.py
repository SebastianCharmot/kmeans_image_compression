from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import copy

def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    dimensions = image.shape
    H = dimensions[0]
    W = dimensions[1]

    centroids_init = np.empty([num_clusters,3])

    for i in range(num_clusters):
        rand_row = random.randint(0,H-1)
        rand_col = random.randint(0,W-1)
        centroids_init[i] =image[rand_row,rand_col]

    return centroids_init


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    -------
    new_centroids : nparray
        Updated centroids
    """

    centroids

    dimensions = image.shape
    H = dimensions[0]
    W = dimensions[1]

    for i in range(max_iter):
        if i % print_every == 0:
            print(f'Currently at iteration {i} of {max_iter}')

        centroid_rgbs = {}

        for row in range(H):
            for col in range(W):
                centroid = np.argmin(np.linalg.norm(centroids - image[row,col], axis=1))
                if centroid in centroid_rgbs:
                    centroid_rgbs[centroid] = np.append(centroid_rgbs[centroid],[image[row,col]],axis=0)
                else:
                    centroid_rgbs[centroid] = np.array([image[row,col]])

        prev_centroids = copy.copy(centroids)

        for k in centroid_rgbs:
            centroids[k] = np.mean(centroid_rgbs[k], axis=0)

        if np.array_equal(prev_centroids, centroids):
            print(f'Converged at iteration {i}')
            return centroids

    return centroids


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    dimensions = image.shape
    H = dimensions[0]
    W = dimensions[1]

    for row in range(H):
        for col in range(W):
            nearest_centroid = np.argmin(np.linalg.norm(centroids - image[row,col], axis=1))
            image[row,col] = centroids[nearest_centroid]

    return image

def plot_centroids(centroids,before):

    # sort by RGB values 
    centroids = centroids.tolist()
    centroids.sort()
    centroids = np.array(centroids)
    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5)) 

    colors = centroids.reshape((16,16,3))

    # plot colors on grid 
    for i in range(16):
        for j in range(16):
            ax1.add_patch(plt.Rectangle((i, j), 0.9, 0.9, facecolor=colors[i, j] / 255))

    ax1.set_xlim((0, 16))
    ax1.set_ylim((0, 16))
    ax1.set_title("Centroids for $k=256$", fontsize=10)
    ax1.set_axis_off()
    
    plt.savefig(fname=f"colors_256_{before}.jpg", transparent=True, format='jpg', bbox_inches='tight', dpi=300)

def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded image with shape: {}'.format(np.shape(image)))

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids = init_centroids(num_clusters, image)
    # plot_centroids(centroids=centroids,before=0)

    # Update centroids 
    update_centroids(centroids,image,max_iter,print_every)

    # plot_centroids(centroids=centroids,before=1)

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title(f'$k={num_clusters}$, run #3')
    plt.axis('off')
    savepath = os.path.join('.', f'compressed_version_k{num_clusters}_run3.png')
    plt.savefig(fname=savepath, transparent=True, format='jpg', bbox_inches='tight', dpi=300)

    print('\nCOMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--large_path', default='cuttlefish.jpg',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=30,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=256,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
