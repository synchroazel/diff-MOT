import torch


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
    return sum(abs(e1 - e2) ** power for e1, e2 in zip(a, b)) ** (1 / power)