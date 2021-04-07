import torch
import Hypercolumn as net
import lib_Unet as lib

Desktop = "Jupyterhub"

########## 1. Obtention des path de la BDD d'image OSM et Optique #############

if Desktop == "Jupyterhub":
    path_bdd_France = "/work/OT/ai4geo/DATA/DATA_MULTIMODAL/France/"
    list_file = lib.get_file_in_folder(path_bdd_France)

elif Desktop == "local":
    path_zone1_local = "C:/Users/natsl/Documents/These/BDD/Zone1"
    list_file = lib.get_file_in_folder(path_zone1_local)

list_OSM_file_France = lib.select_file_name(list_file, 'OSM')
list_optique_file_France = lib.select_file_name(list_file, 'S2')

if Desktop == "Jupyterhub":
    list_OSM_file_France[10].remove(list_OSM_file_France[10][5])
    list_optique_file_France[10].remove([])
    list_OSM_file_France[12].remove(list_OSM_file_France[12][27])
    list_optique_file_France[12].remove([])
    list_OSM_file_France[21].remove([])
    list_optique_file_France[21].remove([])
    list_OSM_file_France = lib.del_xml_file(list_OSM_file_France)

################ 2. Création de la BDD ####################

violet = [155, 89, 182]
bleu = [52, 152, 219]
vert_c = [181, 230, 29]
vert_f = [46, 204, 113]
rouge = [231, 76, 60]
blanc = [236, 240, 241]
orange = [241, 196, 15]

L_ref = [violet, bleu, vert_c, vert_f, rouge, blanc, orange]

#Define the data splits.
nb_train_data = 0.6
nb_val_data = 0.2
nb_test_data = 0.2

percent = 0.08

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

network = net.Hypercolumns()

## 3. Chargement de la BDD

batch_size = 8 #The batch size should generally be as big as your machine can take it.

training_dataset = lib.ImgOptiqueOSM(img_folder["train"], OSM_folder["train"], L_ref)
validate_dataset = lib.ImgOptiqueOSM(img_folder["val"], OSM_folder["val"], L_ref)

print(len(training_dataset))

train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=False)

loss_function = torch.nn.CrossEntropyLoss(ignore_index = 0)

optimizer = torch.optim.SGD(network.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

from torch.autograd import Variable
from tqdm.autonotebook import tqdm #progression bar
import matplotlib.pyplot as plt
import numpy as np
import os

number_epochs = 100
network.to(device)

training_losses = []
validation_losses = []

for epoch in range(number_epochs):
  print("Starting epoch number " + str(epoch))
  
  #Training phase:
  training_loss = 0
  network.train() #indicate to the network that we enter training mode
  
  for i, (inputs, GTs) in enumerate(tqdm(train_loader)):
    #Convert the inputs and GTs to torch Variable (will hold the computation
    #graph) and send them to the computation device (i.e. GPU).
    inputs = Variable(inputs).to(device)
    GTs = Variable(GTs).type(torch.LongTensor).to(device)
    
    #We set the gradients of the model to 0.
    optimizer.zero_grad()
    pred = network(inputs)
    GTs = torch.squeeze(GTs, 1)
    loss = loss_function(pred, GTs)
    #We accumulate the gradients...
    loss.backward()
    #...and we update the parameters according to the gradients.
    optimizer.step()             
    training_loss += loss.cpu().item() / len(train_loader)
    
  print("At epoch #" + str(epoch) + ", loss = " + str(training_loss))
  training_losses.append(training_loss)

  # Enregistrement de la première image de validation pour visualiser l'évolution
  seg_net_display = pred.cpu().detach().numpy()[0]
  seg_net_display = lib.map_to1(seg_net_display)
  seg_net_display = seg_net_display.reshape((256,256))
  plt.figure()
  plt.imshow(seg_net_display, cmap = "hsv")
  plt.title("Segmentation générée à l'epoch " + str(epoch)) 
  if Desktop == "Jupyterhub":
    plt.savefig("../result/hyper_train_image_generee_epoch_"+str(epoch)+".png")
  elif Desktop == "local":
    plt.savefig("C:/Users/natsl/Documents/These/result/zone1/train_image_generee_epoch_"+str(epoch)+".png")
  plt.close()
                
  if epoch == 0:   
    OSMs_display = GTs.cpu().data.numpy()[0]
    OSMs_display = OSMs_display.reshape((256,256))
    plt.figure()
    plt.imshow(OSMs_display, cmap = "hsv")
    plt.title("Image OSM de référence") 
    if Desktop == "Jupyterhub":
      plt.savefig("../result/hyper_train"+str(epoch)+"_ref.png")
    elif Desktop == "local":
      plt.savefig("C:/Users/natsl/Documents/These/result/hyper_train"+str(epoch)+"_ref.png")
    plt.close()
                    
    imgs_display = inputs.cpu().data.numpy()[0]*3
    imgs_display = lib.switch_col_array(imgs_display)
    plt.figure()
    plt.imshow(imgs_display)
    plt.title("Segmentation générée à l'epoch " + str(epoch)) 
    if Desktop == "Jupyterhub":
      plt.savefig("../result/hyper_train_image_optique_epoch_"+str(epoch)+".png")
    elif Desktop == "local":
      plt.savefig("C:/Users/natsl/Documents/These/result/zone1/hyper_train_image_optique_epoch_"+str(epoch)+".png")
    plt.close()
  
  #You need to set the network to eval mode when using batch normalization (to
  #be consistent across evaluation samples, we use mean and stddev computed
  #during training when doing inference, as opposed to ones computed on the
  #batch) or dropout (you want to use all the parameters during inference).
  #In this exercise, we do not use these elements, so just for good practice!
  network.eval()
  
  validation_loss = 0
  #This line reduces the memory by not tracking the gradients. Also to be used
  #during inference.
  with torch.no_grad():
    for i, (inputs, GTs) in enumerate(tqdm(validate_loader)):
      inputs = Variable(inputs).to(device)
      GTs = Variable(GTs).type(torch.LongTensor).to(device)
      GTs = torch.squeeze(GTs, 1)
      pred = network(inputs)
      loss = loss_function(pred, GTs)             
      validation_loss += loss.cpu().item() / len(validate_loader)
    
    
  seg_net_val_display = pred.cpu().detach().numpy()[2]
  seg_net_val_display = lib.map_to1(seg_net_val_display)
  seg_net_val_display = seg_net_val_display.reshape((256,256))
  plt.figure()
  plt.imshow(seg_net_val_display, cmap = "hsv")
  plt.title("Segmentation générée à l'epoch " + str(epoch)) 
  if Desktop == "Jupyterhub":
    plt.savefig("../result/hyper__val_image_generee_epoch_"+str(epoch)+".png")
  elif Desktop == "local":
    plt.savefig("C:/Users/natsl/Documents/These/result/zone1/image_generee_epoch_"+str(epoch)+".png")
  plt.close()
                
  if epoch == 0:   
                    
    OSMs_val_display = GTs.cpu().data.numpy()[2]
    OSMs_val_display = OSMs_val_display.reshape((256,256))
    plt.figure()
    plt.imshow(OSMs_val_display, cmap = "hsv")
    plt.title("Image OSM de référence") 
    if Desktop == "Jupyterhub":
      plt.savefig("../result/hyper_val_ref.png")
    elif Desktop == "local":
      plt.savefig("C:/Users/natsl/Documents/These/result/"+str(epoch)+"_ref.png")
    plt.close()
                    
    imgs_display = inputs.cpu().data.numpy()[2]*3
    imgs_display = lib.switch_col_array(imgs_display)
    plt.figure()
    plt.imshow(imgs_display)
    plt.title("Segmentation générée à l'epoch " + str(epoch)) 
    if Desktop == "Jupyterhub":
      plt.savefig("../result/hyper_val_image_optique_ref.png")
    elif Desktop == "local":
      plt.savefig("C:/Users/natsl/Documents/These/result/zone1/image_optique_epoch_"+str(epoch)+".png")
    plt.close()
            
  print("At epoch #" + str(epoch) + ", validation loss = " + str(validation_loss))
  validation_losses.append(validation_loss)
  
  if epoch > 0:
    plt.figure()
    plt.plot(np.arange(len(training_losses)), training_losses, label = 'Training loss')
    plt.plot(np.arange(len(validation_losses)), validation_losses,  label = 'Validation loss')
    plt.legend()
    if Desktop == "Jupyterhub":
      plt.savefig("../result/hyper_courbe_loss"+str(epoch)+".png")
    elif Desktop == "local":
      plt.savefig("C:/Users/natsl/Documents/These/result/zone1/courbe_loss"+str(epoch)+".png")
    plt.show()
    plt.close()

#Optional, if you want to save your model:
#torch.save(network.state_dict(), os.path.join(base_folder, 'WeightsVaihingen/', 'Hypercolumns_' + str(number_epochs) + "epochs.pth")
  torch.save(network.state_dict(), os.path.join('Hypercolumns_augm_weigths_' + str(number_epochs) + 'epochs.pth'))
    
    
      