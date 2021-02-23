# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 08:14:48 2021

@author: natsl
"""

import torch
import Unet 
import lib_Unet as lib
import train

########## 1. Obtention des path de la BDD d'image SAR et Optique #############

path_zone1_local = "C:/Users/natsl/Documents/These/BDD/Zone1"

list_file_local = lib.get_file_in_folder(path_zone1_local)

list_OSM_file_France = lib.select_file_name(list_file_local, 'OSM')
list_optique_file_France = lib.select_file_name(list_file_local, 'S2')

################ 2. Création de la BDD ####################

noir = [0, 0, 0]
bleu = [52, 152, 219]
vert_c = [181, 230, 29]
vert_f = [46, 204, 113]
rouge = [231, 76, 60]
blanc = [236, 240, 241]
orange = [241, 196, 15]

L_ref = [noir, bleu, vert_c, vert_f, rouge, blanc, orange]

#Define the data splits.
nb_train_data = 0.6
nb_val_data = 0.3
nb_test_data = 0.2

percent = 0.05

img_folder = {}
img_folder["train"] = []
img_folder["val"] = []
img_folder["test"] = []

OSM_folder = {}
OSM_folder["train"] = []
OSM_folder["val"] = []
OSM_folder["test"] = []

nb_imgs_zone = min(len(list_OSM_file_France), len(list_optique_file_France))
sorted_OSM_list = sorted(list_OSM_file_France)
sorted_optique_list = sorted(list_optique_file_France)
for k in range(int(nb_imgs_zone*nb_train_data*percent)):
    OSM_folder["train"].append(sorted_OSM_list[k])  
    img_folder["train"].append(sorted_optique_list[k]) 
for k in range(int(nb_imgs_zone*nb_train_data*percent),int(nb_imgs_zone*(nb_train_data+nb_val_data)*percent)):    
    OSM_folder["val"].append(sorted_OSM_list[k])  
    img_folder["val"].append(sorted_optique_list[k]) 
for k in range(int(nb_imgs_zone*(nb_train_data+nb_val_data)*percent*0.5),int(nb_imgs_zone*percent*0.5)):
    OSM_folder["test"].append(sorted_OSM_list[k])  
    img_folder["test"].append(sorted_optique_list[k]) 

### Melange des deux listes de la même façon
OSM_folder["train"], img_folder["train"] = lib.mix_list(OSM_folder["train"], img_folder["train"])
   
### 3. Chargement de la BDD

batch_size = 2 #The batch size should generally be as big as your machine can take it.

training_dataset = lib.ImgOptiqueOSM(img_folder["train"], OSM_folder["train"], L_ref)
validate_dataset = lib.ImgOptiqueOSM(img_folder["val"], OSM_folder["val"], L_ref)

train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=False)

# x = torch.zeros(5, 3, 256, 256, dtype=torch.float, requires_grad=False)
# y = torch.zeros(1, 2, 256, 256, dtype=torch.float, requires_grad=False)

### 4. Création du réseau

# Initialize generator and discriminator
Net = Unet.Unet()
# print(Generator)
# output_gen = Generator(x)
# print(output_gen)

### 5. Chargement du réseau
# Loss functions
loss_function = torch.nn.CrossEntropyLoss(ignore_index = 0)

# Optimizers
Optimizer = torch.optim.Adam(Net.parameters(), lr=0.001)
# start_epoch = 3
# Generator_weight_path = "C:/Users/natsl/Documents/These/result/" + 'Generator_P2P_Unet_' + str(start_epoch) + "_epochs.pth"
# Discriminator_weight_path = "C:/Users/natsl/Documents/These/result/" + 'Discriminator_P2P_PatchGAN_' + str(start_epoch) + "_epochs.pth"
# Generator.load_state_dict(torch.load(Generator_weight_path))
# Discriminator.load_state_dict(torch.load(Discriminator_weight_path))

number_epochs = 3

train.train_Unet(number_epochs, Net, train_loader, 
              validate_loader, batch_size, Optimizer,
              loss_function)