# LightCSCF

<p float="left"><img src="https://img.shields.io/badge/Python-v3.12.3-green"> <img src="https://img.shields.io/badge/PyTorch-v2.5.1-blue"> <br>

This is the PyTorch implementation for the paper:
>Fang Kai, Yu Zhang, Kaibin Wang, Lei Sang, Yiwen Zhang ["Revisiting Contrastive Learning in Collaborative Filtering via Parallel Graph Filters"](https://doi.org/10.1609/aaai.v40i17.38521)

## Model Illustration

![The figure illustrates the operational process of LightCSCF.](model-figure.png)

## Environment Setting
```python
python == 3.12.3
pytorch == 2.5.1 (cuda:12.4)
scipy == 1.15.3
numpy == 2.3.1
tdqm == 4.65.0
```

## Examples
We used three large-scale datasets: Amazon-book, Tmall and Douban-book. Most of the parameters in LightCSCF are fixed. We only need to adjust the margin hyperparameter `lambda_margin`, temperature coefficient `temperature`, and `lambda_gamma`.

## Examples to Run 
Steps to run the code:
1. In the folder . /configure to configure the LightCSCF.txt file;
2. Run main.py `python main.py` and select the identifier of LightCSCF or specify through the command line:`python main.py --model=LightCSCF`

## Hyperparameter Setting
The best parameters for each dataset are provided as follows: 

Dataset|`lambda_margin`|`lambda_gamma`|`temperature`|`mode`|
|-|-|-|-|-|
Amazon-book|0.7|1|0.2|LightGCN|
Douban-book|0.2|1|0.3|MF|
Tmall|0.4|5|0.2|MF|

## Acknowledgments
This project is built upon the following open-source framework:
- [ID-GRec](https://github.com/BlueGhostYi/ID-GRec)
## Citation
If you find this work helpful, please cite it:
```
@article{Kai_Zhang_Wang_Sang_Zhang_2026, 
title={Revisiting Contrastive Learning in Collaborative Filtering via Parallel Graph Filters}, 
volume={40}, 
url={https://ojs.aaai.org/index.php/AAAI/article/view/38521}, 
DOI={10.1609/aaai.v40i17.38521}, 
number={17}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Kai, Fang and Zhang, Yu and Wang, Kaibin and Sang, Lei and Zhang, Yiwen}, 
year={2026}, 
month={Mar.}, 
pages={14991-14999} 
}
```
