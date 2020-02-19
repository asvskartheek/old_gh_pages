# Understanding Attention Learning (Work in Progress :sweat_smile:)

Most of the tutorials or blog posts on **Attention Learning** that I have come across start with seq2seq problem or the Encoder-Decoder architecture. But the concept of using an attention vector is much more universal and can be applied to a wide variety of issues like:
1. NLP tasks - Named Entity Recognition, Machine Translation etc. [[1]](#references)
2. ML for Systems problems. [[2]](#references)
3. Computer Vision. [[3]](#references)

The characteristic feature among the above three problems is that they all deal with very high dimensional data. NLP tasks deal with vocabularies of the sizes in the range of 100,000. Problems in ML for Systems deal with address space of the order 1e19!!!

Inspiration for an attention-based approach comes from the human eye. Eye concentrates on a part (subject) of the view more than the surroundings. Similarly, here we try to make our model focus more on the useful features.

In this post, we'll take a small example and try to understand the concept of attention learning and its internal working.

## Attention Learning
We assign each feature of the model a weight (attention) ranging from 0 to 1, such that, sum of attentions of all the features is 1. Then we re-feed the network with weighted input vector, which is then passed on to further layers.
![](/images/2020-02-09/attention_diagram.png "Attention Working")

## Problem Statement
Given a data point of **D** dimensions, the task is to classify into 0 or 1. Yep, a simple **Binary Classification**.

## Data Format
**Input**: A D dim vector
**Output**: Single Value either 0/1.

## Model Design
While developing the model, we do not know the function before hand (Obviously!). Now, we want the model to do 2 things:
1. Find the most useful (is *informative* a better word..?) features
2. Find a relationship between the useful features and target.

For the first part, we'll make use of attention learning. To implement this idea, we'll use a layer with **softmax activation** (to ensure that sum of all the attention values is 1)

![](/images/2020-02-09/model_design.png "Model Design")

For the second part, luckily because of **Universal Approximation Theorem** we can use a simple feed forward network. (I will answer why using attention is better than using simple feed forward in the first place in the later section)

## Implementation
```python
def build_attention_model():
    inputs = Input(shape=(input_dims,),name='input_layer')
    attention_vector = Dense(input_dims,activation='softmax',name='attention_vector')(inputs)
    wt_input = multiply([attention_vector,inputs],name='wt_input')
    output = Dense(1,activation='sigmoid',name='output_layer')(wt_input)
    model = Model(inputs=inputs,outputs=output,name='Keras Model')
    return model
```

```python
model = build_attention_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
```

    Model: "Keras Model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_layer (InputLayer)        (None, 10)           0                                            
    __________________________________________________________________________________________________
    attention_vector (Dense)        (None, 10)           110         input_layer[0][0]                
    __________________________________________________________________________________________________
    wt_input (Multiply)             (None, 10)           0           attention_vector[0][0]           
                                                                     input_layer[0][0]                
    __________________________________________________________________________________________________
    output_layer (Dense)            (None, 1)            11          wt_input[0][0]                   
    ==================================================================================================
    Total params: 121
    Trainable params: 121
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None

## Results
```python
model.fit(X,y,epochs=16,batch_size=128,validation_split=0.3)
```

    Train on 70000 samples, validate on 30000 samples
    Epoch 1/16
    70000/70000 [==============================] - 1s 17us/step - loss: 0.6660 - accuracy: 0.5967 - val_loss: 0.6217 - val_accuracy: 0.6911
    .
    .
    .
    TRAINING
    .
    .
    .
    Epoch 16/16
    70000/70000 [==============================] - 1s 15us/step - loss: 0.1937 - accuracy: 0.9250 - val_loss: 0.1974 - val_accuracy: 0.9238

Decent eh..?

## Understanding the inner working
When the first hidden layer i.e, the attention vector is visualised it is very obvious that  features 1 and 2 are the most important.
![](/images/2020-02-09/viz_attention_model.png "Attention Vector")
This information can be used to reduce the number of free variables of the network by removing connections from the remaining features, which improves the model generalizability [[4]](#references). Or you can let the connections acknowledging the contribution of the remaining features in a fuzzy way.

## Why not FNN?


## References
1. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
2. [Neural Hierarchical Sequence Model for Irregular Data Prefetching](https://www.cs.utexas.edu/~akanksha/neural_hierarchical_shi_2019.pdf)
3. [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247)
4. [Generalization and Network Design Strategies](http://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf)