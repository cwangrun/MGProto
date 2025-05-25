# Mixture of Gaussian-distributed Prototypes with Generative Modelling for Interpretable and Trustworthy Image Recognition

This is Pytorch implementation of the paper "[IMixture of Gaussian-distributed Prototypes with Generative Modelling for Interpretable and Trustworthy Image Recognition](https://ieeexplore.ieee.org/document/10982376)", published at IEEE TPAMI 2025.

This code repository was based on ProtoPNet (https://github.com/cfchen-duke/ProtoPNet)

<div align=center>
<img width="500" height="315" src="https://github.com/cwangrun/MGProto/blob/master/figure/intro.png"/></dev>
</div>

**Introduction:** 
Prototypical-part methods (a), such as ProtoPNet, enhance interpretability in image recognition by linking predictions to training prototypes. 
They rely on a point-based learning of prototypes, which have limited representation power and are not suitable to detect Out-of-Distribution (OoD) inputs, 
reducing their decision trustworthiness. 
The point-based learning of prototypes is also unstable and often causes performance drop in the prototype projection step. 
In this work, we present a new generative paradigm to learn prototype distributions (b), termed as Mixture of Gaussian-distributed Prototypes (MGProto).

**Methodology:** In MGProto, we leverage the Gaussian-distributed prototypes to explicitly characterise the underlying data density,
thereby allowing both interpretable image classification and trustworthy recognition of OoD inputs. 
Interestingly, the learning of our Gaussian-distributed prototypes has a natural prototype projection step, effectively addressing the performance degradation issue.

<div align=center>
<img width="830" height="350" src="https://github.com/cwangrun/MGProto/blob/master/figure/method.png"/></dev>
</div>

Additionally, inspired by the ancient legend of Tian Ji’s horse-racing, 
we also present a new and generic prototype mining strategy to enhance prototype learning from abundant less-salient object regions.

<div align=center>
<img width="460" height="380" src="https://github.com/cwangrun/MGProto/blob/master/figure/mining.png"/></dev>
</div>



## Getting Started

**Requirements:** Pytorch, numpy, scipy, cv2, matplotlib, ...

### Prepare datasets

1. Download [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/), and [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/).

4. We primarily employ full images in these datasets and [online augmentations](https://github.com/M-Nauta/ProtoTree/blob/main/util/data.py) for training images.

### Train and test the model

1. Provide data in `data_path, train_dir, test_dir, train_push_dir` in `settings.py`
2. Run `python main.py`, our pre-trained CUB models are given [here]().

### Interpretation with OoD detection

The repository supports online or offline evaluation for interpretable image classification and trustworthy recognition of OoD input. 
This is achieved by computing the overall data probability _**p**_(**x**), 
where in-distribution data (a) yields high _**p**_(**x**) while out-of-distribution input (b) has low _**p**_(**x**).

<div align=center>
<img width="830" height="285" src="https://github.com/cwangrun/MGProto/blob/master/figure/reasoning.png"/></dev>
</div>


### Visualisation results

Prototypes with large prior, which dominate the decision making, are always from high-density distribution regions (in T-SNE) and can localise well the object (bird) parts.
Background prototypes tend to obtain a low prior and come from the low-density distribution regions. 
This finding is used for model compression by pruning the prototypes that hold low prior/importance.

<div align=center>
<img width="830" height="235" src="https://github.com/cwangrun/MGProto/blob/master/figure/visual.png"/></dev>
</div>



## Suggested citation:

```
@article{wang2025mixture,
  title={Mixture of gaussian-distributed prototypes with generative modelling for interpretable and trustworthy image recognition},
  author={Wang, Chong and Chen, Yuanhong and Liu, Fengbei and Liu, Yuyuan and McCarthy, Davis James and Frazer, Helen and Carneiro, Gustavo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```
