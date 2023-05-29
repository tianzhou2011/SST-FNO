import torch
import torch.nn as nn 
from torch.nn import Conv2d,LeakyReLU,BatchNorm2d, ConvTranspose2d,ReLU
import cv2
import numpy as np
#from focal_frequency_loss import FocalFrequencyLoss as FFL
import torch.nn.functional as F

def reg_loss(y,y_hat,temp=0.1):
    '''
    :param y: B,T,X,Y,C
    :param y_hat: B,T,X,Y,C
    :return: 1 value
    '''
    B,T,X,Y,C = y.shape
    delta_y = (y[:,1:,:,:,:]-y[:,:-1,:,:,:]).view(B,T-1,-1)
    delta_yhat = (y_hat[:,1:,:,:,:]-y_hat[:,:-1,:,:,:]).view(B,T-1,-1)
    P_y = F.softmax(delta_y/temp,dim=-1)
    P_yhat = F.log_softmax(delta_yhat/temp,dim=-1)
    kl_loss = nn.KLDivLoss(reduction="mean")
    loss = kl_loss(P_yhat,P_y)
    return loss

def loss_gradient_difference(real_image,generated): # b x c x h x w
    true_x_shifted_right = real_image[:,:,1:,:]# 32 x 3 x 255 x 256
    true_x_shifted_left = real_image[:,:,:-1,:]
    true_x_gradient = torch.abs(true_x_shifted_left - true_x_shifted_right)

    generated_x_shift_right = generated[:,:,1:,:]# 32 x 3 x 255 x 256
    generated_x_shift_left = generated[:,:,:-1,:]
    generated_x_griednt = torch.abs(generated_x_shift_left - generated_x_shift_right)

    difference_x = true_x_gradient - generated_x_griednt

    loss_x_gradient = (torch.sum(difference_x)**2)/2 # tf.nn.l2_loss(true_x_gradient - generated_x_gradient)

    true_y_shifted_right = real_image[:,:,:,1:]
    true_y_shifted_left = real_image[:,:,:,:-1]
    true_y_gradient = torch.abs(true_y_shifted_left - true_y_shifted_right)

    generated_y_shift_right = generated[:,:,:,1:]
    generated_y_shift_left = generated[:,:,:,:-1]
    generated_y_griednt = torch.abs(generated_y_shift_left - generated_y_shift_right)

    difference_y = true_y_gradient - generated_y_griednt
    loss_y_gradient = (torch.sum(difference_y)**2)/2 # tf.nn.l2_loss(true_y_gradient - generated_y_gradient)

    igdl = loss_x_gradient + loss_y_gradient
    return igdl

def gdl_loss(real_images,generateds):
    # calculate the loss for each scale
    scale_losses = []
    for i in range(real_images.shape[1]):
        real_image = real_images[:,i,...]
        generate_image = generateds[:,i,...]
        loss = loss_gradient_difference(real_image,generate_image)
        scale_losses.append(loss)

    # condense into one tensor and avg
    return torch.mean(torch.stack(scale_losses))

def FocalFrequency_Loss(real_images,generateds,loss_weight=1.0, alpha=1.0):
    losses =[]
    for i in range(real_images.shape[1]):
        real_image = real_images[:,i,...]
        generate_image = generateds[:,i,...]
        loss = FFL(loss_weight=1.0, alpha=alpha)(generate_image,real_image)
        losses.append(loss)
    return torch.mean(torch.stack(losses))
            