# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 08:14:48 2021

@author: natsl
"""

import torch
import Unet
import lib_Unet as lib
import train
import time

Desktop = "local"

########## 1. Obtention des path de la BDD d'image OSM et Optique #############

if Desktop == "Jupyterhub":
    path_bdd_France = "../../../../work/DATA/DATASETS/DATA_MULTIMODAL/France/"
    list_file = lib.get_file_in_folder(path_bdd_France)

elif Desktop == "local":
    path_zone1_local = "D:/Documents/These/BDD/Zone1"
    list_file = lib.get_file_in_folder(path_zone1_local)

list_OSM_file_France = lib.select_file_name(list_file, 'OSM')
list_optique_file_France = lib.select_file_name(list_file, 'S2')

if Desktop == "Jupyterhub":
    list_OSM_file_France[9].remove(list_OSM_file_France[9][5])
    list_optique_file_France[9].remove([])
    list_OSM_file_France[11].remove(list_OSM_file_France[11][27])
    list_optique_file_France[11].remove([])
    list_OSM_file_France[20].remove([])
    list_optique_file_France[20].remove([])
    list_OSM_file_France = lib.del_xml_file(list_OSM_file_France)

################ 2. Création de la BDD ####################

violet = [155, 89, 182]
bleu = [52, 152, 219]
vert_c = [181, 230, 29]
vert_f = [46, 204, 113]
rouge = [231, 76, 60]
blanc = [236, 240, 241]
orange = [241, 196, 15]

# L_ref = [violet, bleu, vert_c, vert_f, rouge, blanc, orange]
L_ref = [vert_f]

#Define the data splits.
nb_train_data = 0.6
nb_val_data = 0.2
nb_test_data = 0.2

percent = 0.1

img_folder = {}
img_folder["train"] = []
img_folder["val"] = []
img_folder["test"] = []

OSM_folder = {}
OSM_folder["train"] = []
OSM_folder["val"] = []
OSM_folder["test"] = []

if Desktop == "Jupyterhub":
    for i in range(len(list_OSM_file_France)):
        nb_imgs_zone = min(len(list_OSM_file_France[i]), len(list_optique_file_France[i]))
        list_OSM_file_France[i], list_optique_file_France[i], list_not_pair_OSM_temp, list_not_pair_S2_temp = lib.del_not_pair(list_OSM_file_France[i], list_optique_file_France[i])
        sorted_OSM_list = sorted(list_OSM_file_France[i])
        sorted_optique_list = sorted(list_optique_file_France[i])
        	
        for k in range(int(nb_imgs_zone*nb_train_data*percent)):
            OSM_folder["train"].append(sorted_OSM_list[k])  
            img_folder["train"].append(sorted_optique_list[k]) 
        for k in range(int(nb_imgs_zone*nb_train_data*percent),int(nb_imgs_zone*(nb_train_data+nb_val_data)*percent)):    
            OSM_folder["val"].append(sorted_OSM_list[k])  
            img_folder["val"].append(sorted_optique_list[k]) 
        for k in range(int(nb_imgs_zone*(nb_train_data+nb_val_data)*percent),int(nb_imgs_zone*percent)):
            OSM_folder["test"].append(sorted_OSM_list[k])  
            img_folder["test"].append(sorted_optique_list[k])             
            
elif Desktop == "local":
    nb_imgs_zone = min(len(list_OSM_file_France), len(list_optique_file_France))
    sorted_OSM_list = sorted(list_OSM_file_France)
    sorted_optique_list = sorted(list_optique_file_France)
    for k in range(int(nb_imgs_zone*nb_train_data*percent)):
        OSM_folder["train"].append(sorted_OSM_list[k])  
        img_folder["train"].append(sorted_optique_list[k]) 
    for k in range(int(nb_imgs_zone*nb_train_data*percent),int(nb_imgs_zone*(nb_train_data+nb_val_data)*percent)):    
        OSM_folder["val"].append(sorted_OSM_list[k])  
        img_folder["val"].append(sorted_optique_list[k]) 
    for k in range(int(nb_imgs_zone*(nb_train_data+nb_val_data)*percent),int(nb_imgs_zone*percent)):
        OSM_folder["test"].append(sorted_OSM_list[k])  
        img_folder["test"].append(sorted_optique_list[k]) 
        
        
# ## Melange des deux listes de la même façon
# OSM_folder["train"], img_folder["train"] = lib.mix_list(OSM_folder["train"], img_folder["train"])
   
## 3. Chargement de la BDD

batch_size = 2 #The batch size should generally be as big as your machine can take it.

training_dataset = lib.ImgOptiqueOSM(img_folder["train"], OSM_folder["train"], L_ref)
validate_dataset = lib.ImgOptiqueOSM(img_folder["val"], OSM_folder["val"], L_ref)

print(len(training_dataset))

train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=False)

# train_loader_OSM = {
#         'dataloader': train_loader,
#         }

# torch.save(train_loader_OSM, "../../../../work/users/letheun/These/Unet/Dataset/train_loader_OSM.pth")

# validate_loader_OSM = {
#         'dataloader': validate_loader,
#         }

# torch.save(validate_loader_OSM, "../../../../work/users/letheun/These/Unet/Dataset/validate_loader_OSM.pth")

### load dataset
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_loader = torch.load("../../../../work/users/letheun/These/Unet/Dataset/train_loader_OSM.pth", map_location=torch.device("cpu"))
# print('train dataset loaded')
# validate_loader = torch.load("../../../../work/users/letheun/These/Unet/Dataset/validate_loader_OSM.pth", map_location=torch.device(device))
# print('validate dataset loaded')
### 4. Création du réseau

# Initialize generator and discriminator

Net = Unet.Unet()

### 5. Chargement du réseau
# Loss functions

# weights = [0, 50, 50, 8.5, 3.5, 20, 25, 4]
# class_weights = torch.FloatTensor(weights).cuda()
loss_function = torch.nn.CrossEntropyLoss()#weight=class_weights)

# Optimizers
Optimizer = torch.optim.Adam(Net.parameters(), lr=0.00001)

number_epochs = 1

train.train_Unet(number_epochs, Net, train_loader, 
              validate_loader, batch_size, Optimizer,
              loss_function, Desktop)