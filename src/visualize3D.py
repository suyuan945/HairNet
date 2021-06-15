import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import cv2
from numpy.linalg import inv
import re

proj_dir = '../../../../../data/hairnet'
train_index_path = proj_dir + '/data/index/train.txt'

train_index = []
with open(train_index_path, 'r') as f:
    lines = f.readlines()
    for x in lines:
        train_index.append(x.strip().split(' '))



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

# just like the original gen_RT_matrix, but this one takes in 
# array of length 6 as input rather than txt file
def gen_RT_matrix2(lines):
    R_vec = np.array([float(lines[3]),float(lines[5]), float(lines[4])]).reshape(3, 1)
    T_vec = np.array([float(lines[0]),float(lines[1]), float(lines[2])]).reshape(3, 1)
    R_vec = np.array(R_vec).reshape(3,1)
    T_vec = np.array(T_vec).reshape(3,1)
    R_mat = cv2.Rodrigues(R_vec)[0].reshape(3,3)
    RT_mat = np.hstack((R_mat, T_vec)).reshape(3,4)
    RT_mat = np.vstack((RT_mat, [0,0,0,1])).reshape(4,4)
    return inv(RT_mat)


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

def show3DhairObj(name, index):
    """
    strands: [100, 4, 32, 32]
    mask: [32, 32] bool
    """
    f = open(name, "w+")

    current_index = train_index[index]
    current_convdata_index = re.search('strands\d\d\d\d\d_\d\d\d\d\d_\d\d\d\d\d', str(current_index)).group(0)
    current_RT_mat = gen_RT_matrix(proj_dir+'/data/'+str(current_index[0])+'.txt')
    current_convdata_path = proj_dir+'/convdata/'+str(current_convdata_index)+'.convdata'
    #current_RT_mat = np.dot(current_RT_mat, gen_RT_matrix2([0, 0, 0, 0.5, 0.5, 0.5]))
    strands = get_rendered_convdata(current_convdata_path, current_RT_mat)

    count = 0
    for i in range(32):
        for j in range(32):
            if sum(sum(strands[:, :, i, j])) == 0:
                continue
            strand = strands[:, 0:3, i, j]
            #each strand now has shape (100, 3)
            x = strand[:, 0]
            y = strand[:, 1]
            z = strand[:, 2]
            count += 1
            for k in range(100):
                line = "v " + str(x[k]) + " " + str(y[k]) + " " + str(z[k]) + "\n"
                f.write(line)
    
    for i in range(count):
        line = "l"
        for j in range(1, 101):
            line += " " + str(j + i * 100)
        f.write(line + "\n")

    print("done writing into " + name)


def show3DhairPlot(index):
    """
    strands: [100, 4, 32, 32]
    mask: [32, 32] bool
    """
    f = open("hair.obj", "w+")

    current_index = train_index[index]
    current_convdata_index = re.search('strands\d\d\d\d\d_\d\d\d\d\d_\d\d\d\d\d', str(current_index)).group(0)
    current_RT_mat = gen_RT_matrix(proj_dir+'/data/'+str(current_index[0])+'.txt')
    current_convdata_path = proj_dir+'/convdata/'+str(current_convdata_index)+'.convdata'
    #current_RT_mat = np.dot(current_RT_mat, gen_RT_matrix2([0, 0, 0, 0.5, 0.5, 0.5]))
    strands = get_rendered_convdata(current_convdata_path, current_RT_mat)

    fig = plt.figure(figsize=(40,40))
    ax = fig.add_subplot(111, projection='3d')
    
    avgx, avgy, avgz = 0, 0, 0

    count = 0
    for i in range(32):
        for j in range(32):
            if sum(sum(strands[:, :, i, j])) == 0:
                continue
            strand = strands[:, 0:3, i, j]
            #each strand now has shape (100, 3)
            x = strand[:, 0]
            y = strand[:, 1]
            z = strand[:, 2]
            ax.plot(x, y, z, linewidth=0.2, color='brown')

            avgx += sum(x) / 100
            avgy += sum(y) / 100
            avgz += sum(z) / 100
            count += 1
    
    avgx /= count
    avgy /= count
    avgz /= count
    RADIUS = 0.3  # space around the head
    ax.set_xlim3d([avgx - RADIUS, avgx + RADIUS])
    ax.set_ylim3d([avgy - RADIUS, avgy + RADIUS])
    ax.set_zlim3d([avgz - RADIUS, avgz + RADIUS])
    plt.show()


show3DhairObj("hair10.obj", 10)
show3DhairObj("hair200.obj", 200)
show3DhairPlot(10)
show3DhairPlot(200)