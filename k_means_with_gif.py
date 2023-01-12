import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
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

    # reshape the array into a 8x8 grid
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
    ax1.set_axis_off()

    # read in and plot your image on the right subplot
    compressed_image = update_image(image, centroids)

    ax2.imshow(compressed_image)
    ax2.set_title("Compressed Image", fontsize=10)
    ax2.set_axis_off()

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

# read image 
image = np.copy(mpimg.imread("cuttlefish.jpg"))
print('Loaded image with shape: {}'.format(np.shape(image)))

centroids_init = init_centroids(64, image)

max_iter = 30
print_every = 10

# get centroids 
centroids = update_centroids(centroids_init, image, max_iter, print_every)

# create compressed image using centroids 
image_clustered = update_image(image, centroids)