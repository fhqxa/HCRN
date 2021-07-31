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

## Reference
[Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025)

Learning to Compare: Relation Network for Few-Shot Learning code from [GitHub](https://github.com/floodsung/LearningToCompare_FSL)
