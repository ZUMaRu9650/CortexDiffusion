# Cortex-Diffusion

This repository contains the source code for the MICCAI 2024 paper [Efficient Cortical Surface Parcellation via Full-Band Diffusion Learning at Individual Space](https://papers.miccai.org/miccai-2024/paper/2548_paper.pdf)
![figure1](https://github.com/user-attachments/assets/17451ed6-7b1f-4352-95bc-f416b7b969a6)


## Cite this work

```
@inproceedings{zhu2024efficient,
  title={Efficient Cortical Surface Parcellation via Full-Band Diffusion Learning at Individual Space},
  author={Zhu, Yuanzhuo and Lian, Chunfeng and Li, Xianjun and Wang, Fan and Ma, Jianhua},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={162--172},
  year={2024},
  organization={Springer}
}
```

## Outline

* ```block.py``` contains implementation of _full-band spectral-accelerated spatial diffusion module_ (**FB-SASD**), _spatial gradient_ (**SG**), _vertex-wise multi-layer perceptron_ (**MLP**) and **FB-Diffusion** block
* ```model.py``` contains implementation of **Cortex-Diffusion**
* ```step.py``` contains the functions for running **Cortex-Diffusion** during training, validation, and testing
* ```utils.py``` and ```geometry.py``` mainly contains some useful functions for data processing
* ```dataset.py``` is a subclass of ```torch.utils.data.Dataset```.

## Tips

Each input cortical surface needs to be preprocessed through function ```compute_operators``` in ```geometry.py``` like:
```
_, mass, L, evals, evecs, gradX, gradY = compute_operators(vertices, faces, k)
```
* ```vertices``` stores the 3D vertex coordinates
* ```faces``` stores the vertex indices of each triangle face
* ```k``` is the number of eigenvectors
* ```mass, L, evals, evecs, gradX, gradY``` along with 3D vertex coordinates will be used as model's input, corresponding to line 33 in ```dataset.py``` and line 46 in ```model.py```.

We suggest you perform the above processing for each input mesh in advance and save the results in ```.pt``` format like:
```
d = dict()
d["vertices"] = vertices
d["massvec"] = mass
d["evals"] = evals
d["evecs"] = evecs
d["gradX"] = gradX
d["gradY"] = gradY
torch.save(d, 'your_work_space/surface.pt'))
```

Here is an example of using **Cortex-Diffusion** for cortical surface parcellation:
```
import os
import numpy as np
import torch
import torch.optim as optim
from CortexDiffusion.step import train, val
from CortexDiffusion.dataset import Data
from CortexDiffusion.model import DiffusionNet
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

batch_size = 1  # We suggest you set the batch size to 1, since the number of vertices in different cortical surface may not be the same
device = 'cuda:0'

checkpoint = r"your_work_space\checkpoint_path"
data_path = r"your_work_space\data_path"
folds = os.listdir(root)  # len(folds) = 5 # Assuming your data has been divided into 5 folds in advance and placed in different folders under 'data_path' 
f = open(os.path.join(checkpoint,"logs.txt"), "w+")

for i in range(5):
    # We employed 5-fold cross-validation for performance quantification
    if i == 0:
        l = folds
    else:
        l = folds[i:] + folds[:i]
    test_fold = l[0]
    val_fold = l[1]
    train_fold = l[2:]
    
    train_loss = None
    val_loss = None
    val_dice = None
    val_acc = None
    Loss_train = []
    Loss_val = []

    train_dataset = Data(root=data_path, folds=train_fold, k=200)
    val_dataset = Data(root=data_path, folds=val_fold, k=200)
    test_dataset = Data(root=data_path, folds=test_fold, k=200)

    net = DiffusionNet(in_channels=3, out_channels=32, hidden_channels=128, n_block=4, dropout=True, with_gradient_features=True)
    net.to(torch.device(device))
    save_path_train = os.path.join(checkpoint, 'net_train_' + test_fold + '.pkl')
    save_path_val = os.path.join(checkpoint, 'net_val_' + test_fold + '.pkl')

    ########## train ##########
    print('now test fold is ' + test_fold)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode="min",factor=0.5,patience=2)
    for i in range(100):
        print('now lr = ' + str(optimizer.state_dict()['param_groups'][0]['lr']))
        if optimizer.state_dict()['param_groups'][0]['lr'] < 1e-6:
            break
        loss_train, acc_train = train(net, train_dataset, batch_size, optimizer, i+1, device)
        loss_val, acc_val, dice_val = val(net, val_dataset, batch_size, i+1, device)
        scheduler.step(loss_val)
    
        Loss_train.append(loss_train)
        Loss_val.append(loss_val)
    
        if train_loss is None or loss_train < train_loss:
            train_loss = loss_train
            torch.save(net.state_dict(),save_path_train)
        
        if val_loss is None or dice_val >= val_dice:
            val_loss = loss_val
            val_dice = dice_val
            val_acc = acc_val
            torch.save(net.state_dict(),save_path_val)
    
    net.load_state_dict(torch.load(save_path_val, map_location=device))
    _, _, test_dice = val(net, test_dataset, batch_size, 1, device)
        
    with open(os.path.join(checkpoint,"logs.txt"), 'a+') as f:
        print('now test fold is ' + test_fold + ': train_acc = {:.2f} %'.format(100*acc_train) + ' | val_dice = {:.2f} %'.format(100*val_dice) + ' | test_dice = {:.2f} %'.format(100*test_dice),file=f)
    
    data = dict()
    data['loss_train'] = Loss_train
    data['loss_val'] = Loss_val
    np.save(os.path.join(checkpoint,"experiment_data_" + test_fold + ".dict"),data)
```
Of course, you can also write code according to your personal preferences to run **Cortex-Diffusion**.
