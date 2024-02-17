import os
import numpy as np
from skimage.io import imread, imsave
from skimage.exposure import equalize_adapthist, rescale_intensity


def collectCh0ROI(seg_dir,
                  ch0_dir,
                  blockname,
                  xcoords, 
                  ycoords,
                  zcoords):


    x1, x2  = xcoords[0], xcoords[1]
    y1, y2  = ycoords[0], ycoords[1]       
    zlevels = np.arange(zcoords[0], zcoords[1], 1)   

    # Create the block subfolder names and the jpeg file names for the false-colored training ROIs

    blockdir = os.path.join(seg_dir,
                              '%s_Xpos_%s_%s_Ypos_%s_%s_stack_%s_%s' %
                              (blockname,
                              '{:0>6d}'.format(x1),
                              '{:0>6d}'.format(x2),
                              '{:0>6d}'.format(y1),
                              '{:0>6d}'.format(y2),
                              '{:0>6d}'.format(zcoords[0]),
                              '{:0>6d}'.format(zcoords[1]))
                              )

    blockdir_T = blockdir + "_transpose"
    blockdir_M = blockdir + "_mirror"
    blockdir_F = blockdir + "_flip"

    if not os.path.exists(blockdir):
        print(blockdir)
        os.mkdir(blockdir)
        os.mkdir(blockdir_T)
        os.mkdir(blockdir_M)
        os.mkdir(blockdir_F)

    Segflists = [blockdir + os.sep + '%s_FC_pos%s%s_pos%s%s_' %
                                      (blockname,
                                      x1,
                                      x2,
                                      y1,
                                      y2) +
                        '{:0>6d}.jpeg'.format(z) for z in zlevels]

    SegTlists = [blockdir_T + os.sep + '%s_FC_pos%s%s_pos%s%s_transpose_' %
                                      (blockname,
                                      x1,
                                      x2,
                                      y1,
                                      y2) +
                        '{:0>6d}.jpeg'.format(z) for z in zlevels]

    SegMlists = [blockdir_M + os.sep + '%s_FC_pos%s%s_pos%s%s_mirror_' %
                                      (blockname,
                                      x1,
                                      x2,
                                      y1,
                                      y2) +
                        '{:0>6d}.jpeg'.format(z) for z in zlevels]

    SegFflists = [blockdir_F + os.sep + '%s_FC_pos%s%s_pos%s%s_mirror_' %
                                      (blockname,
                                      x1,
                                      x2,
                                      y1,
                                      y2) +
                        '{:0>6d}.jpeg'.format(z) for z in zlevels]

    chans = [(ch0_dir, 's02')]
    
    flists  = []
    Tflists = []
    Mflists = []
    Fflists = []
    
    # Iterate through channels

    for ch in chans:
    
        blockdir = os.path.join(ch[0],
                                '%s_Xpos_%s_%s_Ypos_%s_%s_stack_%s_%s' %
                                (blockname,
                                '{:0>6d}'.format(x1),
                                '{:0>6d}'.format(x2),
                                '{:0>6d}'.format(y1),
                                '{:0>6d}'.format(y2),
                                '{:0>6d}'.format(zcoords[0]),
                                '{:0>6d}'.format(zcoords[1]))
                                )

        blockdir_T = blockdir + "_transpose"
        blockdir_M = blockdir + "_mirror"
        blockdir_F = blockdir + "_flip"

        flist = [blockdir + os.sep + '%s_%s_pos%s%s_pos%s%s_' %
                                (blockname,
                                ch[1],
                                x1,
                                x2,
                                y1,
                                y2) +
                '{:0>6d}.jpeg'.format(z) for z in zlevels]

        flists.append(flist)

        Tflist = [blockdir_T + os.sep + '%s_%s_pos%s%s_pos%s%s_transpose_' %
                                (blockname,
                                ch[1],
                                x1,
                                x2,
                                y1,
                                y2) +
                '{:0>6d}.jpeg'.format(z) for z in zlevels]

        Tflists.append(Tflist)

        Mflist = [blockdir_M + os.sep + '%s_%s_pos%s%s_pos%s%s_mirror_' %
                                (blockname,
                                ch[1],
                                x1,
                                x2,
                                y1,
                                y2) +
                '{:0>6d}.jpeg'.format(z) for z in zlevels]

        Mflists.append(Mflist)

        Fflist = [blockdir_F + os.sep + '%s_%s_pos%s%s_pos%s%s_flip_' %
                                (blockname,
                                ch[1],
                                x1,
                                x2,
                                y1,
                                y2) +
                '{:0>6d}.jpeg'.format(z) for z in zlevels]

        Fflists.append(Fflist)


    # Read in JPEG images and do false-color
    ROI_blocks = [flists,   Tflists,  Mflists,  Fflists]
    Seg_blocks  = [Segflists, SegTlists, SegMlists, SegFflists]

    return ROI_blocks, Seg_blocks

    # for ROI_block, Seg_block in zip(ROI_blocks, Seg_blocks):
    #     for i in range(len(ROI_block)):
            # if i == 0:
                # ch0  = imread(ROI_block[i])
                # print(ROI_block[i])
                # print(Seg_block[i])
                
                # cyto = imread(ROI_block[1][i])
                # nuc =  FC_rescale(nuc,  1, 10000)
                # cyto = FC_rescale(cyto, 1, 10000)

                # pseudoHE = rapidFalseColor(nuc, cyto, HE_settings['nuclei'], HE_settings['cyto'])
                # imsave(FC_block[i], pseudoHE)

            

