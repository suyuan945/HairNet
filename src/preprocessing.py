'''
Copyright@ Qiao-Mu(Albert) Ren. 
All Rights Reserved.
This is the code to preprocess data of HairNet to train a neural network.
'''

import numpy as np
from numpy.linalg import inv
import cv2

def gasuss_noise(img, mean=0, var=0.001):
    img = np.array(img/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    noisy_img = img + noise
    if noisy_img.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    noisy_img = np.clip(noisy_img, low_clip, 1.0)
    noisy_img = np.uint8(noisy_img*255)
    return noisy_img


# read strandsXXXXX_YYYYY_AAAAA_mBB.txt
# The rotation (Euler angles) and the position of the head. 
# R_x, R_y, R_z, X, Y, Z
# R_vec = (R_x, R_y, R_z), reshape->(3, 1)
# T_vec = (X, Y, Z), reshape->(3, 1)
def gen_RT_matrix(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = lines[0].split(' ')
        R_vec = np.array([float(lines[3]),float(lines[5]), float(lines[4])]).reshape(3, 1)
        T_vec = np.array([float(lines[0]),float(lines[1]), float(lines[2])]).reshape(3, 1)
        R_vec = np.array(R_vec).reshape(3,1)
        T_vec = np.array(T_vec).reshape(3,1)
        R_mat = cv2.Rodrigues(R_vec)[0].reshape(3,3)
        RT_mat = np.hstack((R_mat, T_vec)).reshape(3,4)
        RT_mat = np.vstack((RT_mat, [0,0,0,1])).reshape(4,4)
        return inv(RT_mat)


# read strandsXXXXX_YYYYY_AAAAA_mBB.convdata
# Dimension: 100*4*32*32  
# v[i,0:3,n,m] is the x,y,z position of the ith point on the [n,m]th strand.
# v[i,3,n,m] is a value related to the curvature of that point. 
# if v[:,:,n,m] all equals to 0, it means it is an empty strand.
# x: v[i,3,n,m][0]
def get_rendered_convdata(path, RT_matrix):
    convdata = np.load(path).reshape(100, 4, 32, 32)
    rendered_convdata = convdata
    for i in range(0,32):
        for j in range(0,32):
            if sum(sum(convdata[:,:,i,j])) != 0:
                position = convdata[:,0:3,i,j]
                position = np.hstack((position, np.ones(100).reshape(100,1)))
                position = np.dot(position,RT_matrix).reshape(100,4)
                position[:,0] = position[:,0]/position[:,3]
                position[:,1] = position[:,1]/position[:,3]
                position[:,2] = position[:,2]/position[:,3]
                rendered_convdata[:,0:3,i,j] = position[:,0:3]
    return rendered_convdata


# read strandsXXXXX_YYYYY_AAAAA_mBB.vismap
# v = numpy.load(filename)
# Dimension: 100*32*32
# v[i, n, m] is the visibility of the ith point on the [n,m]th strand. 1 means visible, 0 means invisible. The visibility is computed from the view of the image.
def gen_vis_weight(path, weight_max=10.0, weight_min=0.1):
    vismap = np.load(path)
    weight = vismap
    for i in range(0,32):
        for j in range(0,32):
            for k in range(0,100):
                if vismap[k,i,j] == 1.0:
                    weight[k,i,j] = weight_max
                elif vismap[k,i,j] == 0.0:
                    weight[k,i,j] = weight_min
                else:
                    print('There is something wrong!')
    return weight


