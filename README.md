# LightCSCF

<p float="left"><img src="https://img.shields.io/badge/Python-v3.12.3-green"> <img src="https://img.shields.io/badge/PyTorch-v2.5.1-blue"> <br>

This is the PyTorch implementation for the paper:
>Fang Kai, Yu Zhang, Kaibin Wang, Lei Sang, Yiwen Zhang ["Revisiting Contrastive Learning in Collaborative Filtering via Parallel Graph Filters"]()

## Model Illustration

![The figure illustrates the operational process of LightCSCF.](model-figure.png)

## Environment Setting
```python
python == 3.12.3
pytorch == 2.5.1 (cuda:12.4)
scipy == 1.15.3
numpy == 2.3.1
```

## Examples
We used three large-scale datasets: Amazon-book, Tmall and Douban-book. Most of the parameters in LightCSCF are fixed. We only need to adjust the margin hyperparameter `lambda_margin`, temperature coefficient `temperature`, and `lambda_gamma`.

## Examples to Run 
Steps to run the code:
1. In the folder . /configure to configure the LightCSCF.txt file;
2. Run main.py `python main.py` and select the identifier of MODEL_NAME or specify through the command line:`python main.py --model=Light_CSCF`

## Hyperparameter Setting
The best parameters for each dataset are provided as follows: 

Dataset|`lambda_margin`|`lambda_gamma`|`temperature`|
|-|-|-|-|
Amazon-book|0.7|1|0.2|
Douban-book|0.2|1|0.3|
Tmall|0.4|5|0.2|

## Acknowledgments
This project is built upon the following open-source framework:
- [ID-GRec](https://github.com/BlueGhostYi/ID-GRec)
## Citation
If you find this work helpful, please cite it:
```
```
