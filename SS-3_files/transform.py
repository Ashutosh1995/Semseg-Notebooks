import numpy as np
import torch

from PIL import Image

def colormap_cityscapes(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    #cmap[1,:] = np.array([ 0,0,  0])
    #cmap[0,:] = np.array([128, 64,128])
    #cmap[1:] = np.array([ 70,70,70])
    #cmap[2,:] = np.array([ 220,220,0])
    #cmap[3,:] = np.array([ 107,142,35])
    #cmap[4,:] = np.array([ 70,130, 180])
    #cmap[5,:] = np.array([220, 20,60])
    #cmap[6,:] = np.array([ 0, 0, 142])    
    #cmap[7,:] = np.array([ 0,0,  0])
    
    cmap[0,:] = np.array([128, 64,128]) # road
    cmap[1,:] = np.array([244 ,35, 232]) # sidewalk
    cmap[2,:] = np.array([220, 20, 60]) # pedestrain
    cmap[3,:] = np.array([0, 0, 230]) # motorcycle
    cmap[4,:] = np.array([220,190, 40]) #curb
    cmap[5,:] = np.array([70,70,70]) #building
    cmap[6,:] = np.array([70, 130, 180]) #sky   
    cmap[7,:] = np.array([ 0, 0,  0])    
    return cmap


def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Colorize:

    def __init__(self, n=22):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        print(size)
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        #color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label
            #mask = gray_image == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
