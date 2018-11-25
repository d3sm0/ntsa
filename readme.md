Neural Time Series Analysis
---
## Introduction
Neural Time Series Anaylsis is an attempt to provide an easy to use interface to modern time-series research from the machine learning community. The library features several models, for 1d time-series regression that have been collected thanks to researchers that published their code and hackers that tried to reproduce them.

---
## Features

1. Recurrent models can use the Input Attention mechanism as developed in the DARNN paper, to increate interpretability in case of high-dimensional feature space.
2. (Conditional) Neural Processes in the regression setting, can provide mean and variance for each estimate.
3. Soft Dynamic Time Warping for time-series alignment

---
## Datasets
This library comes with the following datasets:
- [SML 2010](https://archive.ics.uci.edu/ml/datasets/SML2010)
- [NASDAQ 100](http://cseweb.ucsd.edu/~yaq007/NASDAQ100_stock_data.html)
- [BTC]

Pickle file can be found [here](https://drive.google.com/open?id=0B3B22Hd5PMxSaVpsYmRHU2ZJYWc).

To adapt a new dataset, please check utils.dataset_utils.py.

---
## Models

- Dense
- RNN
- Seq2seq
- [DARNN](https://arxiv.org/pdf/1704.02971.pdf)
- [Neural Process](https://arxiv.org/abs/1807.01622)
- [Conditional Neural Process](https://arxiv.org/abs/1807.01613)
- [Soft Dynamic Time Warping](https://arxiv.org/abs/1703.01541)

---
## Install

For Pip users:

### Pip
```bash
git clone github.com/d3sm0/ntsa.git
cd ntsa
pip install -r requirements.txt
```
This library also requires to install the [soft-dtw](https://github.com/mblondel/soft-dtw) loss function. 

### Docker

If you are used to docker than maybe this can work better:
```bash
make docker
make dev
```
---
## Usage

```bash
python main.py --model seq2seq --loss sdtw  --mode train --data_path data/
```

---
## Remark
This repo is still under refinement and I'm planning to move to eager mode in the upcoming weeks. Nevertheless is usable for early experiments, there for Fork it and hack with it :) 
Pull request are welcome, and if you need help in making it work, just open an issue.

I warmly advice to read the original paper of the model or loss function that you are planning to use, before moving in the experimental part.
