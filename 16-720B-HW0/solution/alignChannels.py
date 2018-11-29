import numpy as np

def best_offset(im1, im2):
    H = im1.shape[0]
    W = im1.shape[1]

    best_offset = None
    min_cost = H*W*300
    for dy in range(-30, 31):
        for dx in range(-30, 31):
            shift_im1 = np.roll(im1, (dy, dx), (0, 1))
            if dy > 0:
                shift_im1[:dy, :] = 0
            else:
                shift_im1[dy:, :] = 0
            if dx > 0:
                shift_im1[:, :dx] = 0
            else:
                shift_im1[:, dx:] = 0

            cost = np.sum(np.abs((im2 - shift_im1) ** 2))

            if cost < min_cost:
                min_cost = cost
                best_offset = shift_im1

    return best_offset

def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""


    green_shift = best_offset(green, red)
    blue_shift = best_offset(blue, red)

    return np.stack((red, green_shift, blue_shift), axis=2)
