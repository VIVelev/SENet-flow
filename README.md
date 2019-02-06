# SENet-flow
Squeeze-and-Excitation Network (SENet) implementation in [TensorFlow](https://www.tensorflow.org)

***You can find the original paper [here](https://arxiv.org/pdf/1709.01507.pdf)*** <br>
written by Jie Hu, Li Shen, Gang Sun

If you want to see ***the original author's code***, please refer to this [link](https://github.com/hujie-frank/SENet)

## Requirements
 - Python 3.x
 - Tensorflow 1.x

## Idea
### What is SENet?
<div align="center">
  <img src="https://github.com/VIVelev/SENet-flow/blob/master/figures/SE-pipeline.jpg">
</div>
<p align="center">
  Figure 1: Diagram of a Squeeze-and-Excitation building block.
</p>
<br>

### How do you integrate it in existing powerful architectures? (Inception Network, ResNet)
<div align="center">
   <img src="https://github.com/VIVelev/SENet-flow/blob/master/figures/SE-Inception-module.jpg" width="420">
  <img src="https://github.com/VIVelev/SENet-flow/blob/master/figures/SE-ResNet-module.jpg"  width="420">
</div>
<p align="center">
  Figure 2: Schema of SE-Inception and SE-ResNet modules.
</p>
<br>

### Hyperparameters
 - Reduction Ratio - controls the bottleneck size (the number of units in the bottleneck dense layer).
<div align="center">
   <img src="https://github.com/VIVelev/SENet-flow/blob/master/figures/Reduction-Ratio.jpg" width="420">
</div>
<p align="center">
  Figure 3: Different choices for the Reduction Ratio hyperparameter and the consequent results.
</p>

## Why should you use Squeeze-and-Excitation Netwroks
### State-of-the-art performace on ILSVRC 2017 (ImageNet 2017 dataset)
<div align="center">
   <img src="https://github.com/VIVelev/SENet-flow/blob/master/figures/State-of-the-art.jpg" width="420">
</div>
<p align="center">
  Figure 4: State-of-the-art performace on ILSVRC 2017.
</p>
