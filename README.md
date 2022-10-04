# Good-DA-in-KD
### [Project](https://mingsun-tse.github.io/Good-DA-in-KD/) | [ArXiv](https://arxiv.org/abs/2012.02909) | [PDF](https://arxiv.org/pdf/2012.02909.pdf) 

<div align="center">
    <a><img src="figs/merl.png"  height="120px" ></a>
    &nbsp
    <a><img src="figs/smile.png"  height="100px" ></a>
</div>

This repository is for our NeurIPS 2022 paper:
> **[What Makes a "Good" Data Augmentation in Knowledge Distillation -- A Statistical Perspective](https://mingsun-tse.github.io/Good-DA-in-KD/)** \
> [Huan Wang](http://huanwang.tech/)<sup>1,2</sup>, [Suhas Lohit](https://suhaslohit.github.io/)<sup>2</sup>, [Michael Jones](https://www.merl.com/people/mjones)<sup>2</sup>, [Yun Fu](http://www1.ece.neu.edu/~yunfu/)<sup>1</sup> \
> <sup>1</sup>Northeastern University <sup>2</sup>MERL \
> Work done when Huan was an intern at MERL.

**[TL;DR]** We study the question what makes a good data augmentation (DA) in knowledge distillation (KD) A proposition from a statistical perspective is proposed, suggesting a good DA should minimize the stddev of the teacher's mean probability. Per the proposition, a metric to measure the "goodness" of DA is intriduced, which works well empirically.

<div align="center">
    <a><img src="figs/overview.svg"  width="700" ></a>
    </br>
    Investigation overview of applying a *stronger* DA in KD in our paper. The question is: By what, we mean the "stronger" DA?
</div>

## Reproducing Our Results

### 0. Download the code
```
git clone git@mingsun-tse.com/Good-DA-in-KD.git -b master
cd Good-DA-in-KD
```


### 1. Set up (original) data
```bash
```

### 2. Set up environment with Anaconda
- `conda create --name Good-DA-in-KD python=3.9.6`
- `conda activate R2L`
- `pip install -r requirements.txt` (We use torch 1.9.0, torchvision 0.10.0)

### 3. Quick start: test our trained models
- Download models:
```
sh scripts/download_R2L_models.sh
```

- Run
  
 

## Results
TOADD

## Acknowledgments
In this code we heavily rely on the wonderful [code of CRD](https://github.com/HobbitLong/RepDistiller). Great thanks to them! We also greatly thank the anounymous NeurIPS'22 reviewers for the constructive comments to help us improve the paper.

## Reference

If our work or code helps you, please consider to cite our paper. Thank you!
```BibTeX
@inproceedings{wang2022what,
  author = {Huan Wang and Suhas Lohit and Michael Jones and Yun Fu},
  title = {What Makes a "Good" Data Augmentation in Knowledge Distillation -- A Statistical Perspective},
  booktitle = {NeurIPS},
  year = {2022}
}
```


