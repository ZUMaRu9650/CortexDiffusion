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
_, mass, L, evals, evecs, gradX, gradY = compute_operator(vertices, faces, k)
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
