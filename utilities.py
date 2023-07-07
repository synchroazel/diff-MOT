import numpy as np
import torch
import os
import sys
def create_subtracks(track_directory:str, frames: int) -> None:
    """ This function is used to create subtracks of a MOT track. To avoid memory cluttering, it creates symbolic links in the subfolders"""

    images = os.listdir(track_directory)
    images = [x for x in images if x[-4] == '.']

    subtrack_base = os.path.normpath(os.path.join(track_directory,"subtrack_"))

    j = 0

    for i, image in enumerate(images):

        image_path = os.path.normpath(os.path.join(track_directory, image))

        if i % frames == 0:
            # create subfolder
            subfolder = subtrack_base + str(j)
            j += 1
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)

        # create symbolic link
        try:
            os.symlink(src=image_path, dst=os.path.normpath(os.path.join(subfolder,image)))
        except FileExistsError:
            pass

def get_best_device():
    """Function used to get the best device between cuda, mps and cpu"""
    if torch.cuda.is_available():
        print("[INFO] Using CUDA.")
        return torch.device("cuda")
    elif torch.has_mps:
        print("[INFO] Using MPS.")
        return torch.device("mps")
    else:
        print("[INFO] No GPU found. Using CPU.")
        return torch.device("cpu")

def minkowski_distance(a:list, b:list, power=2):
    """

    :param a:
    :param b:
    :param power:
    :return:
    """
    if power < 1:
        raise Exception("Power of Minkowski distance should not be less than 1")
    if a is None or b is None:
        raise Exception("While calculating the distance, one of the two elements was none")
    # return np.float16(sum(abs(e1 - e2) ** power for e1, e2 in zip(a, b)) ** (1 / power))
    return sum(abs(e1 - e2) ** power for e1, e2 in zip(a, b)) ** (1 / power)