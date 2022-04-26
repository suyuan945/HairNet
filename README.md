# HairNet

# Setup

* Data can be downloaded from BaiduYun(https://pan.baidu.com/s/1CtWSRARsdUX_xO-2IjTM1w, password: ezh6)
or generate data using https://github.com/papagina/HairNet_DataSetGeneration.

# Training

Training by running this command with list of arguments below:

```bash
python train.py --data ./data/ --epochs 10 --lr 1e-4
```

List of Arguments
| Args         | Type  |                                                              |
|--------------|-------|--------------------------------------------------------------|
| --epoch      | int   | Number of epoch                                              |
| --batch_size | int   | Number of batch size                                         |
| --lr         | float | Learning rate                                                |
| --lr_step    | int   | Number of epoch to reduce 1/2 lr                             |
| --data       | str   | Path to ./data/                                              |
| --save_dir   | str   | Path to save trained weights                                 |
| --weight     | str   | Load weight from path                                        |
| --test_step  | int   | If `test_step` != 0, test each `n` step after train an epoch |

*Notes*: Hyperparameters of original HairNet.  
* Epoch: 500
* Batch size: 32
* Learning rate: 1e-4
* Learning rate step: 10 epochs
* Optimization: Adam  
