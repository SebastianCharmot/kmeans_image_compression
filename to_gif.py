import glob
from PIL import Image
import re

""" Creates a GIF of all the png images in a folder """

def make_gif(frame_folder):
 
    frames = []

    # sort by name so proper order is preserved 
    imgs = sorted(glob.glob("{}/*.png".format(frame_folder)))
    imgs.sort(key=lambda f: int(re.sub('\D', '', f)))

    print(imgs)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    
    # Save into a GIF file that loops forever
    frames[0].save('cuttlefish.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=300, loop=0)
    
if __name__ == "__main__":
    make_gif("to_gif_6")