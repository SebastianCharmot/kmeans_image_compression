import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.colors
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
        rand_row = random.randint(0,H)
        rand_col = random.randint(0,W)
        centroids_init[i] = image[rand_row,rand_col]

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

    losses = []
    iterations = []

    centroids

    dimensions = image.shape
    H = dimensions[0]
    W = dimensions[1]

    for i in range(1,max_iter+1):
        iterations.append(i)
        
        if i % print_every == 0:
            print(f'Currently at iteration {i} of {max_iter}')
            # make_visualization(image,centroids,i)

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

        cur_loss = make_visualization(image,centroids,i,losses,iterations)
        # losses.append(cur_loss)

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

    im = copy.copy(image)

    for row in range(H):
        for col in range(W):
            nearest_centroid = np.argmin(np.linalg.norm(centroids - im[row,col], axis=1))
            im[row,col] = centroids[nearest_centroid]

    return im


def make_visualization(image, centroids, cur_iteration, losses, iterations):

    colors = centroids

    # reshape the array into a 3x3 grid
    colors = colors.reshape((8,8,3))

    # create a figure with two subplots
    # , width_ratios=[1,1.6]
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # plot the colors on the left subplot
    for i in range(8):
        for j in range(8):
            ax1.add_patch(plt.Rectangle((i, j), 0.9, 0.9, facecolor=colors[i, j] / 255))

    ax1.set_xlim((0, 8))
    ax1.set_ylim((0, 8))

    ax1.set_title("Centroids", fontsize=10)
    # ax1.grid(visible=False)
    ax1.set_axis_off()

    # read in and plot your image on the right subplot
    compressed_image = update_image(image, centroids)

    ax2.imshow(compressed_image)
    ax2.set_title("Compressed Image", fontsize=10)
    # ax2.spines.right.set_visible(False)
    # ax2.spines.top.set_visible(False)
    ax2.set_axis_off()

    # err = np.sum((image - compressed_image) ** 2)
    # cur_mean_loss = err/(image.shape[0] * compressed_image.shape[1])

    # print(image)
    # print(type(image))
    # print(compressed_image)
    # print(type(compressed_image))


    # difference = image - compressed_image
    # cur_mean_loss = np.mean(difference)
    # print(cur_mean_loss)
    # print()

    cur_mean_loss = np.mean(np.linalg.norm(image-compressed_image,axis=2))

    losses.append(cur_mean_loss)

    print(iterations)
    print(losses)
    # ax3.plot(iterations,losses, color='blue', linewidth=2)

    # # add a title to the third subplot
    # ax3.set_title("Average Pixel-Wise Loss", fontsize=10)
    # ax3.set_xlabel("Iteration")
    # ax3.set_xlim(left=1,right=30)
    # ax3.set_ylim(bottom=0)

    # plt.subplots_adjust(bottom=0.15, wspace=0.05)
    # plt.subplots_adjust(wspace=-0.5)
    plt.suptitle("Iteration {}".format(cur_iteration))

    # show the plot
    plt.savefig('to_gif_6/iteration_{}.png'.format(cur_iteration), dpi=300, bbox_inches='tight')
    # plt.show()

    return cur_mean_loss

    # reshape the array into a 3x3 grid
    colors = centroids.reshape((3, 3, 3))

    # create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # plot the colors on the left subplot
    ax1.imshow(colors)

    # read in and plot your image on the right subplot
    # image = plt.imread('your_image.jpg')
    ax2.imshow(image)

    # show the plot
    plt.show()
    plt.savefig('iteration_{}.png'.format(cur_iteration),transparent=True, dpi=300)

    return 

    colors = np.asarray(centroids, dtype='uint8')

    image = update_image(image, centroids)

    # Create a Numpy array of RGB values
    # colors = np.array([[131, 27, 28],
    #                 [69, 55, 31],
    #                 [131, 149, 88]])

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    # Add a 3x3 grid of colored rectangles to the first subplot
    ax1.imshow(colors)
    ax1.axis('off')

    # Add the image to the second subplot
    ax2.imshow(image)
    ax2.axis('off')

    plt.savefig('iteration_{}.png'.format(cur_iteration),transparent=True, dpi=300)

    return 

    colors = centroids
    print(colors)

    colors = [['red', (255,0,0)], 
          ['green', (0,255,0)], 
          ['blue', (0,0,255)],
          ['yellow', (255,255,0)],
          ['cyan', (0,255,255)],
          ['magenta', (255,0,255)],
          ['black', (0,0,0)],
          ['white', (255,255,255)],
          ['gray', (128,128,128)]]

    # Load an image and get its dimensions
    image = update_image(image,centroids)
    height, width, channels = image.shape

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    # Add a 3x3 grid of colored rectangles to the first subplot
    for i, color in enumerate(colors):
        # print(np.append(color,1))
        # color = matplotlib.colors.to_rgba_array(np.asarray(color, dtype='uint8'), alpha=1)
        # rgba = np.append(color, 1)
        # rgba = color
        # color = tuple(np.asarray(rgba, dtype='uint8'))
        # print(color)
        row = i // 3
        col = i % 3
        ax1.add_patch(plt.Rectangle((col, row), 1, 1, color=color[1]))
        ax1.text(col+0.5, row+0.5, str(color[0]), ha='center', va='center', fontsize=10)
    ax1.axis('off')

    # Add the image to the second subplot
    ax2.imshow(image)
    ax2.axis('off')

    # Save the plot as a PNG file with high definition
    plt.savefig('iteration_{}.png'.format(cur_iteration),transparent=True, dpi=300)

    return

    # Define a list of 9 colors and their corresponding RGB values
    colors = centroids

    # Load an image and get its dimensions
    # image = plt.imread('image.png')
    height, width, channels = image.shape

    image = update_image(image,centroids)

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    # Add a 3x3 grid of colored rectangles to the first subplot
    for ax, color in zip(ax1.flat, colors):
        ax.add_patch(plt.Rectangle((0,0), 1, 1, color=color))
        ax.set_title(str(color))

        # if color is in the following format: [['red', (255,0,0)], ['green', (0,255,0)], ...
        # ax.add_patch(plt.Rectangle((0,0), 1, 1, color=color[1]))
        # ax.set_title(color[0])
        ax.axis('off')

    # Add the image to the second subplot
    ax2.imshow(image)
    ax2.axis('off')

    # Save the plot as a PNG file with high definition
    plt.savefig('iteration_{}.png'.format(cur_iteration), dpi=300)

# main 

# read image 
image = np.copy(mpimg.imread("cuttlefish.jpg"))
print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))

centroids_init = init_centroids(64, image)

max_iter = 30
print_every = 10

# get centroids 
centroids = update_centroids(centroids_init, image, max_iter, print_every)

# create compressed image using centroids 
image_clustered = update_image(image, centroids)