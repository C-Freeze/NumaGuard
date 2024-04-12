from pprint import pprint

def get_pins():
    """
    Create 4 arrays of 0-9
    Shuffle 3 of them, to get a derangment (The first does not need to change)
    Concatenate the arrays
    Transpose the array
    return the array
    """

    import random
    import numpy as np

    # Create 4 arrays of 0-9
    a = np.array(range(10))
    b = np.array(range(10))
    c = np.array(range(10))
    d = np.array(range(10))

    # Shuffle 3 of them, to get a derangment
    random.shuffle(b)
    random.shuffle(c)
    random.shuffle(d)

    # Stack the arrays
    e = np.stack((a, b, c, d))

    return e

if __name__ == '__main__':
    pprint(get_pins())