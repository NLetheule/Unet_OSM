# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 08:16:13 2021

@author: natsl
"""

import os
import gdal
import numpy as np
from skimage import io
import sys, time, random
import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from torch.utils.data import Dataset
import torchvision.transforms as T

def get_file_in_folder(folder):
    """
        Liste récursivement le contenu des sous-répertoires
    """
    list_file = []
    for f in os.listdir(folder):
        if os.path.isdir(folder+'/'+f): # si f est un dossier
            list_file.append(get_file_in_folder(folder+'/'+f))
        else :
            list_file.append(folder+'/'+f) 
    
    return(list_file)

def select_file_name(list_file, word):
    list_selected = []
    for file in list_file:
        if type(file) is list:
            list_selected.append(select_file_name(file, word))
        elif str(file).find(word) != -1:
                list_selected.append(file)
    return list_selected

class ImgOptiqueOSM(Dataset):
    def __init__(self, img_folder, OSM_folder, L_ref, patch_size=256):
        self.imgs = []
        self.OSMs = []

        #This will convert the numpy array to a tensor
        conversion = T.ToTensor()
        overlap = patch_size 

        for img_index in range(0,len(img_folder)):
            sys.stdout.write("Working on image " + str(img_index+1)+"/"+str(len(img_folder)) + "\n") 
            sys.stdout.flush() 
            time.sleep(random.random())
            #Load the tile and the corresponding SAR truth.
            if assert_same_tile(OSM_folder[img_index], img_folder[img_index]):
                img = normalize_imgs(img_rgb(io.imread(img_folder[img_index])))
                img = nan_to_zero(img)
                OSM_file = OSM_folder[img_index]
                OSM_raster = gdal.Open(OSM_file)
                OSM_band1 = OSM_raster.GetRasterBand(1)
                OSM_array1 = OSM_band1.ReadAsArray()
                OSM_band2 = OSM_raster.GetRasterBand(2)
                OSM_array2 = OSM_band2.ReadAsArray()
                OSM_band3 = OSM_raster.GetRasterBand(3)
                OSM_array3 = OSM_band3.ReadAsArray()
                OSM = np.zeros((len(OSM_array1),len(OSM_array1[0])))
                OSM_array = np.array([OSM_array1, OSM_array2, OSM_array3])
                OSM = OSM_label2(L_ref, OSM_array)
#                 for k in range(len(OSM_array1)):
#                     for l in range(len(OSM_array1[0])):
#                         #OSM_fin[k,l] = nearest_neighbor(L_ref, OSM_array[:,k,l])
#                         OSM[k,l] = OSM_label(L_ref, OSM_array[:,k,l])
                if OSM.shape == (1024,1024) and np.max(img) != 0:
                    for i in np.arange(patch_size//2, img.shape[0] - patch_size // 2 + 1, overlap):
                        for j in np.arange(patch_size//2, img.shape[1] - patch_size // 2 + 1, overlap):
                              #Crop the image and the ground truth into patch around (i,j) and save
                              #them in self.imgs and self.SARs arrays.
                              #For the image, note that we are taking the three channels (using ":")
                              #for the 3rd dimension, and we do the conversion to tensor.
                              self.imgs.append(conversion(img[i - patch_size//2:i + patch_size // 2, j - patch_size // 2:j + patch_size // 2,:]))
                              self.OSMs.append(conversion(OSM[i - patch_size//2:i + patch_size // 2, j - patch_size // 2:j + patch_size // 2]))
 
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        
        img = self.imgs[idx].float()
        OSM = self.OSMs[idx].long()

        return img, OSM
    
def normalize_img(image):
    out = image - np.nanmean(image)
    image_std = np.nanstd(image)
    
    if image_std != 0:
        out /= image_std
        
    out = np.clip(out, -3, 1)
    
    return out

def normalize_imgs(img):

    img = img - img.min()
    img = img/img.max()
    img = img.astype(float)
    
    return img

# def normalize_imgs(img):

#     img = img - 0
#     img = img/22769
#     img = img.astype(float)
    
#     return img

def img_rgb(img):
    
    img_rgb = np.zeros(img.shape)
    img_rgb[:,:,0] = img[:,:,2]
    img_rgb[:,:,1] = img[:,:,1]
    img_rgb[:,:,2] = img[:,:,0]
    
    return img_rgb

def mix_list(list_SAR, list_img):
    
    array = [list_SAR, list_img]
    array = np.transpose(array)
    np.random.shuffle(array)
    array = np.transpose(array)
    list_SAR = array[0]   
    list_img = array[1]
    
    return list_SAR, list_img

def nan_to_zero(image):
    out = image.copy()
    out[np.isnan(out)] = 0
    return out

def OSM_label(L_ref, value):
    index = 0
    for i in range(len(L_ref)):
        if (L_ref[i][0] == value[0]) & (L_ref[i][1] == value[1]) & (L_ref[i][2] == value[2]): 
            index = i+1
            return index
    return index

def OSM_label2(L_ref, OSM_array):
    OSM_fin = np.zeros((len(OSM_array[0]),len(OSM_array[0][0])))
    for i in range(len(L_ref)):
        index = i + 1
        filtre = np.ones(OSM_array.shape)
        filtre[0] *= L_ref[i][0]
        filtre[1] *= L_ref[i][1]
        filtre[2] *= L_ref[i][2]
        test = (filtre == OSM_array)
        test1 = test[0] & test[1] & test[2]
        OSM_fin = OSM_fin + test1*index
    return OSM_fin


# Fonction qui met les différentes couche de softmax des labels en une seule carte de segmentation
def map_to1(seg_map):
    map_fin = np.zeros((len(seg_map[0]), len(seg_map[0][0])))

    for i in range(len(seg_map[0])):
        for j in range(len(seg_map[0][0])):
            L = seg_map[:,i,j]
            map_fin[i,j] = np.argmax(L)
            
            
    return map_fin
        
def del_xml_file(list_file):
    for i in range(len(list_file)):
        list_temp = list_file[i]
        for j in list_temp:
            if j.endswith('.xml') or j.endswith('.xml.tiff'):    
                list_temp.remove(j)
        list_file[i] = list_temp
    
    return list_file

def switch_col_array(img):
    array = np.zeros((len(img[0][0]),len(img[0]),len(img)))
    array[:,:,0] = img[0]
    array[:,:,1] = img[1]
    array[:,:,2] = img[2]
#     array = np.array([R, G, B])
    return array 

def tile_number(file_OSM, file_S2):
    OSM = []
    S2 = []
    for i in range(len(file_OSM)):
        OSM.append([x for x in re.findall(r'\d+', file_OSM[i])[-2:]])
    for i in range(len(file_S2)):
        S2.append([x for x in re.findall(r'\d+', file_S2[i])[-2:]])
    return OSM, S2

def del_not_pair(file_OSM, file_S2):
    OSM, S2 = tile_number(file_OSM, file_S2)
    not_pair_OSM = [ x for x in OSM if x not in S2]
    not_pair_S2 = [ x for x in S2 if x not in OSM]
    list_not_pair_S2 = []
    list_not_pair_OSM = []
    if len(not_pair_OSM) != 0:
        for i in range(len(not_pair_OSM)):
            del_file = [file for file in file_OSM if re.findall(r'\d+', file)[-2:] == not_pair_OSM[i] ]
            list_not_pair_OSM.append(del_file[0])
            file_OSM.remove(del_file[0])
    if len(not_pair_S2) != 0: 
        for i in range(len(not_pair_S2)):
            del_file = [file for file in file_S2 if re.findall(r'\d+', file)[-2:] == not_pair_S2[i] ]
            list_not_pair_S2.append(del_file[0])
            file_S2.remove(del_file[0])
    return file_OSM, file_S2, list_not_pair_OSM, list_not_pair_S2

def assert_same_tile(file_OSM, file_S2):
    tile_OSM = [x for x in re.findall(r'\d+', file_OSM)[-2:]]
    tile_S2 = [x for x in re.findall(r'\d+', file_S2)[-2:]]
    return(tile_OSM == tile_S2)

def display_OSM(OSM, title, save, name_save):
    violet = [155, 89, 182]
    bleu = [52, 152, 219]
    vert_c = [181, 230, 29]
    vert_f = [46, 204, 113]
    rouge = [231, 76, 60]
    blanc = [236, 240, 241]
    orange = [241, 196, 15]
    labels = ["noir", "violet", "bleu", "vert_c", "vert_f", "rouge", "blanc", "orange"]
    cmap = colors.ListedColormap([ 'black', np.divide(violet, 256), np.divide(bleu, 256), np.divide(vert_c, 256), np.divide(vert_f, 256), np.divide(rouge, 256), np.divide(blanc, 256), np.divide(orange, 256)])
    boundaries = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    plt.figure()
    plt.imshow(OSM, cmap=cmap, norm=norm)
    plt.colorbar(label = labels )
    plt.title(title) 
    if save == True : 
        plt.savefig(name_save)
    plt.close()