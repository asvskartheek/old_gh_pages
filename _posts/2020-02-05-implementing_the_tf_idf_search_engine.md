# Implementing the TF-IDF Search Engine

A straightforward way to make a search engine is using a vector space model (VSM).
In this post, let’s look at the concept of VSM and implement it in Python 3 using **numpy**, **pandas** and
**scikit-learn**.
<br />

The basic idea behind VSM is to represent text in the form of a vector. Although there are many efficient 
and sophisticated approaches to render text in a vectorised format, in this post, we will consider a naive 
way. Fun fact, there is a sub-field of AI and ML called [**Knowledge Representation**](https://en.wikipedia.org/wiki/Knowledge_representation_and_reasoning)
that deals with efficient ways of representing text.

![](/images/2020-02-05/vsm.png "Vector Space Model")

## Representing Text
Consider the following sentences,
