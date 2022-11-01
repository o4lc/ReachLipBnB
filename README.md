# bnb-nn-verification

This repository constains the code for ReachLipBnB paper which will appear as a conference paper at ICRA 2023.
This framework is a branch and bound based method for neural network reachability analysis introduced by Taha Entesari, Sina Sharifi and Mahyar Fazlyab.

# Installation Requirements
```
pip install torch
pip install cvxpy
pip install polytope
```

# Usage
As a simple example, you could use the `.json` files in `./Config` folder in order to rune some test examples.
In order to do so, you can set the parameter `filename` in `run.py` and the run the code:
```
python3 run.py
```
