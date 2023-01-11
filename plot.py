import numpy as np
import matplotlib.pyplot as plt

# create a 3x3 grid of the nine colors in your numpy array
colors = np.array([[ 74.43825682 , 96.61069303 ,113.25059862],
                [ 14.86106777 , 41.25089246 ,101.58695825],
                [ 50.8908597  , 76.28994887 ,110.79458293],
                [ 11.12968516, 102.11769115 ,190.96026987],
                [173.48267676 , 33.73283716 , 10.32112152],
                [211.69273335 , 98.16712139 , 46.18718057],
                [ 40.11930374 , 69.6854142  ,118.38255972],
                [ 67.96714611 , 16.00542325 ,  9.89098483],
                [172.24694877 ,135.26887378 ,114.83030037],
                [ 17.28813559 ,105.83274704 , 49.2811001 ],
                [ 18.53523917 , 55.46006227 ,116.499349  ],
                [ 13.6124031  , 30.29564699 , 13.12128801],
                [102.90772721 , 65.48415855 , 72.81153076],
                [ 46.98775056 , 57.96687082  ,28.33723088],
                [239.36821589 ,222.06262583, 202.69389591],
                [  6.7180462  ,  4.75093601  , 4.63210645]])

# reshape the array into a 3x3 grid
colors = colors.reshape((4,4,3))

# create a figure with two subplots
# , width_ratios=[1,1.6]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# plot the colors on the left subplot
for i in range(4):
    for j in range(4):
        ax1.add_patch(plt.Rectangle((i, j), 0.9, 0.9, facecolor=colors[i, j] / 255))

ax1.set_xlim((0, 4))
ax1.set_ylim((0, 4))

ax1.set_title("Centroids", fontsize=10)
# ax1.grid(visible=False)
ax1.set_axis_off()

# read in and plot your image on the right subplot
image = plt.imread('cosmo.jpg')
ax2.imshow(image)
ax2.set_title("Compressed Image", fontsize=10)
# ax2.spines.right.set_visible(False)
# ax2.spines.top.set_visible(False)
ax2.set_axis_off()

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# plot the data on the third subplot
ax3.plot(x, y, color='red', linewidth=2)

# add a title to the third subplot
ax3.set_title("Total Loss", fontsize=10)

ax3.set_xlabel("Iteration")
ax3.set_xlim(left=1,right=30)

# plt.subplots_adjust(bottom=0.15, wspace=0.05)
# plt.subplots_adjust(wspace=-0.5)
plt.suptitle("Iteration_{}".format(1))

# show the plot
plt.savefig('to_gif/iteration_{}.png'.format(50), dpi=300, bbox_inches='tight')
plt.show()
