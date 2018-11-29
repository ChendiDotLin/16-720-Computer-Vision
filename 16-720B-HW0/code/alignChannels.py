def alignChannels(red, green, blue):
    import numpy as np
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""


    [row,col] = red.shape
    #initialize min value
    mingreen = np.sum(np.square(green - red))
    minblue = np.sum(np.square(blue - red))

    for ioffset in range(-30,31): # negative number shift up, positive down
        g1 = np.roll(green, ioffset, axis=0)
        b1 = np.roll(blue, ioffset, axis=0)

        # if ioffset <0 row+ioffset to end = 0
        # if ioffset >0 0 to ioffset = 0

        if ioffset < 0:
            g1[row + ioffset:, :] = 0
            b1[row + ioffset:, :] = 0
        else:
            g1[:ioffset, :] = 0
            b1[:ioffset, :] = 0

        for joffset in range(-30,31): # negative number shift left, positive right
            g2 = np.roll(g1,joffset,axis = 1)
            b2 = np.roll(b1, joffset, axis=1)


            # if joffset <0 col+ioffset to end = 0
            # if joffset >0 0 to joffset = 0
            if joffset < 0:

                g2[:,col+joffset:] = 0
                b2[:,col+joffset:] = 0
            else:
                g2[:,:joffset] = 0
                b2[:,:joffset] = 0

            # find sum
            gdiff = np.sum(np.square(g2 - red))
            bdiff = np.sum(np.square(b2 - red))

            if (gdiff <= mingreen):
                mingreen = gdiff
                gshift = [ioffset, joffset]
                goutput = g2

            if (bdiff <= minblue):
                minblue = bdiff
                bshift = [ioffset, joffset]
                boutput = b2

            #print(mingreen,minblue)
    # goutput = np.roll(green,gshift[0],axis = 0)
    # goutput = np.roll(goutput,gshift[1],axis = 0)
    # boutput = np.roll(blue, bshift[0], axis=0)
    # boutput = np.roll(boutput, bshift[1], axis=1)
    # if gshift[0] < 0:
    #     goutput[row + gshift[0]:, :] = 0
    # else:
    #     goutput[:gshift[0], :] = 0
    # if bshift[0] < 0:
    #     boutput[row + bshift[0]:, :] = 0
    # else:
    #     boutput[:bshift[0], :] = 0
    #
    # if gshift[1] < 0:
    #     goutput[:, col + gshift[1]:] = 0
    # else:
    #     goutput[:, :gshift[1]] = 0
    # if bshift[1] < 0:
    #     boutput[:, col + bshift[1]:] = 0
    # else:
    #     boutput[:, :bshift[1]] = 0


    print(gshift)
    print(bshift)


    return np.dstack((red,goutput,boutput))
