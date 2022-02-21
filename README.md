# HCRN
HCRN full name is Hierarchical Few-shot Learning Based on Coarse- and Fine-grained Relation Network.

# Datasets
[Omniglot](https://github.com/floodsung/LearningToCompare_FSL/blob/master/datas/omniglot_28x28.zip)

[miniImageNet](https://github.com/floodsung/LearningToCompare_FSL/tree/master/datas/miniImagenet)

# Code
omniglot 5way 1 shot:

```
python HC_RN_Omniglot_5to5_s1b19_L2.py -w 5 -s 1 -b 19 
```

omniglot 5way 5 shot:

```
python HC_RN_Omniglot_5to5_s5b15_L2.py -w 5 -s 5 -b 15 
```

omniglot 20way 1 shot:

```
python HC_RN_Omniglot_20to20_s1b10_L2.py -w 20 -s 1 -b 10 
```

omniglot 20way 5 shot:

```
python HC_RN_Omniglot_20to20_s5b5_L2.py -w 5 -s 5 -b 5 
```

mini-Imagenet 5 way 1 shot:

```
python HC_RN_miniImageNet_5to5_s1b15_L2.py -w 5 -s 1 -b 15 
```

mini-Imagenet 5 way 5 shot:

```
python HC_RN_miniImageNet_5to5_s5b10_L2.py -w 5 -s 5 -b 10 
```

you can change -b parameter based on your GPU memory.

## Few-shot classification results
Experimental results on few-shot classification datasets.
The optimizer is Adam with initial learning rate 10−3 and reduced the learning rate by half for every 10,000 episodes.
We report average results with 2,000 randomly sampled episodes.

<table>
  <tr>
    <td>datasets</td>
    <td colspan="2" align="center">Ominiglot</td>
  </tr>
  <tr>
    <td>setups</td>
    <td>5-way 1-shot</td>
    <td>5-way 5-shot</td>
    <td>20-way 1-shot</td>
    <td>20-way 5-shot</td>
  </tr>
  <tr>
    <td>MAML</td>
    <td align="center">99.1</td>
    <td align="center">99.9</td>
    <td align="center">96.1</td>
    <td align="center">99.1</td>
  </tr>
  <tr>
    <td>Relation Network</td>
    <td align="center">99.8</td>
    <td align="center">99.9</td>
    <td align="center">97.8</td>
    <td align="center">99.2</td>
  </tr>
  <tr>
    <td>HCRN</td>
    <td align="center">99.9</td>
    <td align="center">99.9</td>
    <td align="center">98.0td>
    <td align="center">99.7</td>
  </tr>
</table>

<table>
  <tr>
    <td>datasets</td>
    <td colspan="2" align="center">miniImageNet</td>
  </tr>
  <tr>
    <td>setups</td>
    <td>5-way 1-shot</td>
    <td>5-way 5-shot</td>
  </tr>
  <tr>
    <td>LEO</td>
    <td align="center">61.84</td>
    <td align="center">77.71</td>
  </tr>
    <tr>
    <td>E3BM</td>
    <td align="center">64.20</td>
    <td align="center">80.54</td>
  </tr>
   <tr>
    <td>HCRN</td>
    <td align="center">70.67</td>
    <td align="center">78.06</td>
  </tr>
</table>

## Reference
[Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025)

Learning to Compare: Relation Network for Few-Shot Learning code from [GitHub](https://github.com/floodsung/LearningToCompare_FSL)
