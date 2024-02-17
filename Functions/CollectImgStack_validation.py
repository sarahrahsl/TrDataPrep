import os
import numpy as np
import skimage as sk
from pathos.multiprocessing import ProcessingPool


def writeImages(imagePair):
    """
    Parallelized image writing.

    Parameters
    ----------

    imagePair : tuple

        fname : str or pathlike
            Filepath to save image.

        img : 2D numpy array
            Image to write to disk.

    Returns
    -------

    """
    from skimage.io import imsave
    fname = imagePair[0]
    img = imagePair[1]
    imsave(fname, img)
    return


def __preProcess(image):
    """

    Step 1) CLAHE via skimage equalize_adapthist: enhances contrast to make
    full use of the 16 bit dynamic range remaps histogram to 8 bit.

    Parameters
    ----------

    image : 3D numpy array, dtype = uint16
        3D image to undergram histogram equalization and conversion to 8-bit.

    Returns
    -------

    equalized_image : 3D numpy array, dtype = uint8
        3D image with equalized histogram mapped to 8-bit range.

    """
    from skimage.exposure import equalize_adapthist
    import numpy as np
    
    image = np.clip(image,0,3000)

    # set clahe kernel size to be 1/4 image size
    kernel_size = np.asarray([image.shape[0]//4,
                              image.shape[1]//4,
                              image.shape[2]//4])

    # equalize histogram and convert to 8 bit
    equalized_image = equalize_adapthist(image,
                                         kernel_size=kernel_size,
                                         clip_limit=0.01)*255

    # return equalized image as np.uint8 dtype
    return equalized_image.astype(np.uint8)


def collectImgStackFused(f,
                    saveroot,
                    block_name,
                    zcoords,
                    xcoords,
                    ycoords,
                    orient,
                    CLAHE=True,
                    hiclip_val  = 1500, # target channel (ck5, ck8, pgp)
                    lowclip_val = 100,  # target channel (ck5, ck8, pgp)
                    nuc_clip_low   = 100,   # nuc channel
                    nuc_clip_high  = 5000,  # nuc channel
                    cyto_clip_low  = 100,   # cyto channel
                    cyto_clip_high = 3000,
                    res = 1): # cyto channel
    """
    Method for reading in chunks of h5 data with specified coordinates,
    preprocessing the data as necessary, and saving it as 8-bit zstacks,
    in a sequential order.


    Parameters
    ----------

    f : h5 File
        h5.File object with data to chunk

    saveroot : str or pathlike
        root directory for training data

    block_name : str
        Sample name for directory name in training dir

    zcoords : tuple
        Tuple containing z coordinates of image to read in. Should always
        result in a stack of 100 images

    xcoords : tuple
        lateral coordinates for image to read in. Should be roughly 700 for
        oxford

    ycoords : tuple
        lateral coordinates for image to read in. Should be roughly 700 for
         oxford

    res : int
        The resolution to read.


    Returns
    -------

    """

    sch1 = 's01'
    sch2 = 's02'
    sch0 = 's00'
    target_chan = sch2
    
    chans = [(sch1, 'ch1'),
             (sch2, 'ch0'),
             (sch0, 'ch2')]

    x1, x2  = xcoords[0], xcoords[1]
    y1, y2  = ycoords[0], ycoords[1]
    zlevels = np.arange(zcoords[0], zcoords[1], 1)

    tile_str = sch0+sch1+sch2

    # loop through channels
    for ch in chans:

        # create directory structure
        chan_dir = os.path.join(saveroot, ch[1])
        if not os.path.exists(chan_dir):
            os.mkdir(chan_dir)

        blockdir = os.path.join(chan_dir,
                                '%s_Xpos_%s_%s_Ypos_%s_%s_stack_%s_%s' %
                                (block_name,
                                 '{:0>6d}'.format(x1),
                                 '{:0>6d}'.format(x2),
                                 '{:0>6d}'.format(y1),
                                 '{:0>6d}'.format(y2),
                                 '{:0>6d}'.format(zcoords[0]),
                                 '{:0>6d}'.format(zcoords[1]))
                                )
        

        if not os.path.exists(blockdir):
            print(blockdir)
            os.mkdir(blockdir)

        # generate filenames
        flist = [blockdir + os.sep + '%s_%s_pos%s%s_pos%s%s_' %
                                     (block_name,
                                      ch[0],
                                      x1,
                                      x2,
                                      y1,
                                      y2) +
                 '{:0>6d}.jpeg'.format(z) for z in zlevels]
        
        # read in image chunk for channel
        print('reading img', ch[0])
        r = str(res)
        if orient == 1:
            img = f['t00000'][ch[0]][r]['cells'][x1:x2,
                                                zcoords[0]:zcoords[1],
                                                y1:y2].astype(np.uint16)
            img = np.moveaxis(img, 0, 1)
        else:
            img = f['t00000'][ch[0]][r]['cells'][zcoords[0]:zcoords[1],
                                                x1:x2,
                                                y1:y2].astype(np.uint16)
        # do preprocessing on target channel
        if (CLAHE == True) and (ch[0] == target_chan):
            img = __preProcess(img)
            
        elif (CLAHE==False) and (ch[0] == target_chan):
            img = np.clip(img, lowclip_val, hiclip_val)
            img = sk.exposure.rescale_intensity(img, out_range='uint8')

        # remap nulcear or eosin 16 bit images to 8 bit range
        else:
            if ch[1] == 'ch2':
                img = sk.exposure.rescale_intensity(
                                np.clip(img,cyto_clip_low, cyto_clip_high), 
                                out_range='uint8')
            
            else:
                img = sk.exposure.rescale_intensity(
                                np.clip(img,nuc_clip_low,nuc_clip_high), 
                                out_range='uint8')

        # generate list of two entry tuples: filename & corresponding image
        pairedList = [(flist[i], img[i]) for i in range(len(flist))]

        # print('paired mlist', len(pairedList_mirror))

        # save images in a multiprocessed way
        # with ProcessingPool(ncpus=os.cpu_count()) as p:
        #     p.map(writeImages, pairedList)
        #     p.map(writeImages, pairedList_T)
        #     p.map(writeImages, pairedList_mirror)
        #     p.map(writeImages, pairedList_flip)

        # save images in a sequential way
        for i in range(len(pairedList)):
            paired = pairedList[i]
            writeImages(paired)

    return