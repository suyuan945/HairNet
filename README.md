# HairNet
 This is the implementation of [HairNet: Single-View Hair Reconstruction using Convolutional Neural Networks](https://arxiv.org/abs/1806.07467, 'HairNet') using pytorch by rqm 2019.  
 Copyright@ Qiao-Mu(Albert) Ren. All Rights Reserved.  


## Requirement
* pytorch  
* opencv  
* matplotlib

## Preparation
* Before you train or test HairNet, you must make sure all convdata files are in the subfolder 'convdata' and other data files(including vismap, txt, exr, png) are in the subfolder 'data'.  
* If you don't have traning data(including convdata, vismap, txt, exr, png), you can download from BaiduYun(https://pan.baidu.com/s/1CtWSRARsdUX_xO-2IjTM1w, password: ezh6) or generate data using the code from https://github.com/papagina/HairNet_DataSetGeneration.  
* Besides, there is a subfolder 'index' in folder 'data'. The files in 'index' are list.txt, train txt and test.txt. The content in the above files is the index, such as 'strands00025_00409_10000_v0'. If you choose to generate data using the code from https://github.com/papagina/HairNet_DataSetGeneration, you should generate list.txt, train.txt and test.txt by yourself.

## Train
* **Note: this implementation only accounts for position loss and curvature loss.**  
* The arguments of training are mode and project path.  
* An example bash to run this programme: ```python src/main.py --mode train --path '/home/albertren/Workspace/HairNet/HairNet-ren'```   
* Weights of Neural Network will be saved in the subfolder 'weight' per 5 epochs.  
* Log will be saved in Log.txt per 100 batches.  
* Hyperparameters are all setted according to the paper of HairNet.  
Epoch: 100 (origin: 500)  
Batch size: 32  
Learning rate: 1e-4(divided by 2 per 10 epochs, we change this setting according to our experiment)  
Optimization: Adam  

## Test
* The arguments of training are mode, project path and weight path.  
* An example bash to run this programme: ```python src/main.py --mode test --path '/home/albertren/Workspace/HairNet/HairNet-ren' --weight '/home/albertren/Workspace/HairNet/HairNet-ren/weight/000001_weight.pt'```  

## Acknowledgement
Thank [ZZM](https://github.com/TneitaP) for helping me train thie neural network on GPU machine and give me help in daily research.

