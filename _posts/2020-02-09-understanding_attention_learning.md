# Understanding Attention Learning (Work in Progress :sweat_smile:)

Most of the tutorials or blog posts on **Attention Learning** that I have come across start with seq2seq problem or the Encoder-Decoder architecture. But the concept of using an attention vector is much more universal and can be applied to a wide variety of issues like:
1. NLP tasks - Named Entity Recognition, Machine Translation etc. [1](#references)
2. ML for Systems problems. [2](#references)
3. Computer Vision. [3](#references)

The characteristic feature among the above three problems is that they all deal with very high dimensional data. NLP tasks deal with vocabularies of the sizes in the range of 100,000. Problems in ML for Systems deal with address space of the order 1e19!!!

Inspiration for an attention-based approach comes from the human eye. Eye concentrates on a part (subject) of the view more than the surroundings. Similarly, here we try to make our model focus more on the useful features.

In this post, we'll take a small example and try to understand the concept of attention learning and its internal working.

## Attention Learning
We assign each feature of the model a weight ranging from 0 to 1, such that, sum of weights of all the features is 1. Then we re-feed the network with weighted input vector, which is then passed on to further layers.

## Problem Statement
Given a data point of **D** dimensions, the task is to classify into 0 or 1. Yep, a simple **Binary Classification**.

## Data Format
**Input**: A D dim vector
**Output**: Single Value either 0/1.

## Model Design
While developing the model, we do not know the function before hand (Obviously!). Now, we want the model to do 2 things:
1. Find the most useful (is *informative* a better word..?) features
2. Find a relationship between the useful features and target.

For the first part, we'll make use of attention learning
![](/images/2020-02-09/model_design.png "Model Design")

$f(X) = \frac{X[1]+X[2]}{2}$, is the ideal function needed to be modelled.

## References
1. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
2. [Neural Hierarchical Sequence Model for Irregular Data Prefetching](https://www.cs.utexas.edu/~akanksha/neural_hierarchical_shi_2019.pdf)
3. [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247)