import numpy as np
import torch as t
import numpy as np

def HorizentalFlip(ct_image):
    '''
    wheels for honrizentalFlip
    :param ct_image: CxHxW  C is batch of CT image, H is height,W is width
    :return: Fliped image
    '''
    image = np.zeros_like(ct_image)
    for batch in range(ct_image.shape[0]):
        for h_index in range(ct_image.shape[1]):
            image[batch][h_index] = ct_image[batch][ct_image.shape[1]-h_index-1]
    return image

def VerticalFlip(ct_image):
    '''
    wheels for verticalFlip
    :param ct_image: CxHxW  C is batch of CT image, H is height,W is width
    :return: Fliped image
    '''
    image = np.zeros_like(ct_image)
    for batch in range(ct_image.shape[0]):
        for v_index in range(ct_image.shape[2]):
            image[batch,:,v_index] = ct_image[batch,:,ct_image.shape[2]-v_index-1]
    return image

def RandomFlip(ct_image):
    '''
    wheels for RandomFlip
    :param ct_image: CxHxW  C is batch of CT image, H is height,W is width
    :return: Fliped image
    '''
    state = np.random.randint(4)
    if state==0:
        return ct_image
    elif state==1:
        return HorizentalFlip(ct_image)
    elif state==2:
        return VerticalFlip(ct_image)
    elif state==3:
        image = HorizentalFlip(ct_image)
        return VerticalFlip(image)
