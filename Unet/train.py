# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 08:15:16 2021

@author: natsl
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 9:08:48 2020

@author: natsl
"""

import torch
from torch.autograd import Variable
import time
import datetime
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import lib_Unet as lib


def train_net(optimizer, OSMs, fake_OSM, loss_function, batch_size):
    
    optimizer.zero_grad()
    OSMs = torch.squeeze(OSMs)
    # Calculate error and backpropagate
    error = loss_function(fake_OSM, OSMs)
    error.backward(retain_graph=True)
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

def validate_net(optimizer, OSMs, fake_OSM, loss_function, batch_size):
    # Calculate error 
    OSMs = torch.squeeze(OSMs)
    error = loss_function(fake_OSM, OSMs)
    return error

def display_result_per_epoch():

        

    return

def train_Unet(number_epochs, Net, train_loader, 
              validate_loader, batch_size, optimizer,
              loss_function):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    Net.to(device)
    
    marqueur = 0
    nb_display_result = 1
    start_time = time.time()
    training_losses = []
    validation_losses = []
    
    training_loss = 0
    
    validation_loss = 0
    
    for epoch in range(number_epochs):
        print("Starting epoch number " + str(epoch+1))
      
        Net.train() 
      
        for i, (imgs, OSMs) in enumerate(train_loader):
            # #Convert the imgs and SARs to torch Variable (will hold the computation
            # #graph).
            imgs = Variable(imgs).to(device)
            OSMs = Variable(OSMs).to(device)
            seg_net = Net(imgs.float())  
            
            ##### Train Generator #########
            loss = train_net(optimizer, OSMs, seg_net, loss_function, batch_size)     
            
            training_loss += loss.cpu().item()/len(train_loader.dataset.imgs)
            # Display Progress every few batches
        if epoch % nb_display_result == 0: 
            
            print("At epoch #" + str(epoch+1) +"/"+ str(number_epochs) 
                  + ", training loss = " + str(training_loss))
            training_losses.append(training_loss)
            training_loss = 0
  
            #Validation phase:
            Net.eval()
          
            #This line reduces the memory by not tracking the gradients. Also to be used
            #during inference.
            with torch.no_grad():
                for j,  (imgs_val, OSMs_val) in enumerate(validate_loader):
                    imgs_val = Variable(imgs_val).to(device)
                    OSMs_val = Variable(OSMs_val).to(device)
                  
                    seg_net_val = Net(imgs_val.float())
                    
                    loss = validate_net(optimizer, OSMs_val, seg_net_val, loss_function, batch_size)     
                    
                    validation_loss += loss.cpu().item() / len(validate_loader.dataset.imgs)
                    
                    
                if marqueur == 0:
                    OSMs_val_display = OSMs_val.cpu().data.numpy()[0]
                    OSMs_val_display = OSMs_val_display.reshape((256,256))
                    plt.figure()
                    plt.imshow(OSMs_val_display, cmap = "gray")
                    plt.title("Image OSM de référence") 
                    plt.savefig("C:/Users/natsl/Documents/These/result/zone1/image1"+str(j)+"_ref.png")
                marqueur = 1
                # Enregistrement de la première image de validation pour visualiser l'évolution
                seg_net_val_display = seg_net_val.cpu().detach().numpy()[0]
                seg_net_val_display = lib.map_to1(seg_net_val_display)
                seg_net_val_display = seg_net_val_display.reshape((256,256))
                plt.figure()
                plt.imshow(seg_net_val_display, cmap = "gray")
                plt.title("Segmentation générée à l'epoch " + str(epoch)) 
                plt.savefig("C:/Users/natsl/Documents/These/result/zone1/image_generee_epoch_"+str(epoch)+".png")
            
            # Determine approximate time left
            epoch_done = epoch + 1
            epoch_left = number_epochs - epoch_done
            time_left = datetime.timedelta(seconds=epoch_left * (time.time() - start_time)/ epoch_done)
    
            print("At epoch #" + str(epoch+1) + ", validation loss = " +
                  str(validation_loss)  + " " + str(time_left) + " hours left" + "\n")
            validation_losses.append(validation_loss)
            training_loss = 0
            
      
        if epoch > 0:
            
            plt.figure()
            plt.title("Courbes des loss durant l'apprentissage")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.plot(np.arange(len(training_losses)), training_losses, label = 'Generator training loss')
            plt.plot(np.arange(len(validation_losses)), validation_losses, label = 'Generator validation loss')
            plt.legend()
            plt.savefig("C:/Users/natsl/Documents/These/result/zone1/courbe_loss"+str(epoch)+".png")
            plt.show()
            
            
    
        # Save network
        state_net = {
        'epoch': epoch,
        'state_dict': Net.state_dict(),
        'optimizer': optimizer.state_dict()
        }
        torch.save(state_net, "C:/Users/natsl/Documents/These/result/Generator_P2P_Unet_" + str(epoch) + "_epochs.pth")
        