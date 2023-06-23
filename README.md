# JKO-Flow
Pytorch implementation of JKO flow which as a continuous normalizing flow framework that eliminates the need for tuning hyperparameters.

## Associated Publication

Taming hyperparameter tuning in continuous normalizing flows using the JKO scheme

Paper: [https://ojs.aaai.org/index.php/AAAI/article/view/17113](https://www.nature.com/articles/s41598-023-31521-y)

Supplemental: [https://arxiv.org/abs/2006.00104](https://arxiv.org/abs/2211.16757)

Please cite as
    
@article{vidal2023taming,
  title={Taming hyperparameter tuning in continuous normalizing flows using the JKO scheme},
  author={Vidal, Alexander and Wu Fung, Samy and Tenorio, Luis and Osher, Stanley and Nurbekyan, Levon},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={4501},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

## Set-up

Install all the requirements:
```
pip install -r requirements.txt 
```

## Toy problems

Toy problem type and hyperparameters may be selected in the EvaluateToy_JKOflow.py and python evaluateToy_OTflow.py.  In order to function properly, the same problem type and hyperparameters must be selected in both files.  

Train a toy example
```
python3 EvaluateToy_JKOflow.py
```

Evalaute toy model and plot results 
```
python3 EvaluateToy_JKOflow.py
``
