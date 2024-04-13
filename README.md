# <center> PKU-AIGI-500K
**DCC2024: Xunxu Duan, Hongbin Liu, Li Zhang, [Chuanmin Jia](http://www.jiachuanmin.site/index.html)**
**JETCAS 2024: Xunxu Duan, Siwei Ma, Hongbin Liu, [Chuanmin Jia](http://www.jiachuanmin.site/index.html)**

[Project Page](https://duanener.github.io/PKU-AIGI-500K/) | [Paper 1](https://duanener.github.io/PKU-AIGI-500K/) | [Paper 2](https://ieeexplore.ieee.org/document/10493034)

:hammer:This repository is the official repository of the PKU-AIGI-500K benchmark.

:smiley:If this project helps you, please fork, watch, and give a star to this repository.

<!-- # Dataset -->

![example](./static/images/example.jpg) 


![why](./static/images/why.png) 

## Statistics
|            |      Train    |  Validation  |    Test     |   Size  |
|:----------:|:-------------:|:------------:|:-----------:|:-----------:|
|   SD2.1B   |**202,265**(512 $\times$ 512)|**40**(512 $\times$ 512)|**40**(768 $\times$ 768)|~82G|
|   SD2.1    |**203,242**(768 $\times$ 768)|**40**(768 $\times$ 768)|**40**(1024 $\times$ 1024)|~189G|
|  SDXL1.0B  |**105,554**(1024 $\times$ 1024)|**20**(1024 $\times$ 1024)|**40** $\times$ **2**(1280 $\times$ 1280)|~158G|
|   MJ5.2    |**13,240**(1024 $\times$ 1024)|**10**(1024 $\times$ 1024)|**25**(2048 $\times$ 1024)|~20G|
|    MOD     |**3,670**(1408 $\times$ 640)|**5**(1408 $\times$ 640)|**25**(2112 $\times$ 960)|~5.5G|

**Total**: 105k+ texts, 528k+ images, ~455G.


# Citation
:smiley:If you find our repository useful for your research, please consider citing our paper:

```
@ARTICLE{10493034,
  author={Duan, Xunxu and Ma, Siwei and Liu, Hongbin and Jia, Chuanmin},
  journal={IEEE Journal on Emerging and Selected Topics in Circuits and Systems}, 
  title={PKU-AIGI-500K: A Neural Compression Benchmark And Model for AI-Generated Images}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Image coding;Codecs;Circuits and systems;Measurement;Image synthesis;Image quality;Integrated circuit modeling;Image Compression;AIGI;image Feature;text-to-image alignment and subjective evaluation},
  doi={10.1109/JETCAS.2024.3385629}}
```
