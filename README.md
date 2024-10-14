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

* ```block.py``` implementation of _full-band spectral-accelerated spatial diffusion module_ (**FB-SASD**), _spatial gradient_ (**SG**), _vertex-wise multi-layer perceptron_ (**MLP**) and **FB-Diffusion** block
* ```model.py``` implementation of **Cortex-Diffusion**
* ```step.py``` contains the functions for running **Cortex-Diffusion** during training, validation, and testing
* ```utils.py``` and ```geometry.py``` mainly contains some useful functions for data processing
* ```dataset.py``` a subclass of ```torch.utils.data.Dataset```.
