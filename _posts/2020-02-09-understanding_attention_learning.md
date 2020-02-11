# Understanding Attention Learning (Work in Progress :sweat_smile:)

Most of the tutorials or blog posts on **Attention Learning** that I have come across start with seq2seq problem or the Encoder-Decoder architecture. But the concept of using an attention vector is much more universal and can be applied to a wide variety of issues like:
1. NLP tasks - Named Entity Recognition, Machine Translation etc. [1](#references)
2. ML for Systems problems. [2](#references)
3. Computer Vision. [3](#references)

The characteristic feature among the above three problems is that they all deal with very high dimensional data. NLP tasks deal with vocabularies of the sizes in the range of 100,000. Problems in ML for Systems deal with address space of the order 1e19!!!

Inspiration for an attention-based approach comes from the human eye. Eye concentrates on a part (subject) of the view more than the surroundings. Similarly, here we try to make our model focus more on the useful features.

In this post, we'll take a small example and try to understand the concept of attention learning and its internal working.

## Problem Statement
Given a data point of **D** dimensions, the task is to classify into 0 or 1. Yep, a simple **Binary Classification**.

## Dataset
**Input**: Random initialising D dim vector
**Output**: Random 0/1. A little twist, the fourth dimension of the input vector is the same as the output value. So, we want the model to approximate the following function.

$$
f(X) = X[4]
$$
This is the ideal function needed to be modelled.

## Model Design
![](/images/2020-02-09/model_design.png "Model Design")

## References
1. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
2. [Neural Hierarchical Sequence Model for Irregular Data Prefetching](https://www.cs.utexas.edu/~akanksha/neural_hierarchical_shi_2019.pdf)
3. [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247)