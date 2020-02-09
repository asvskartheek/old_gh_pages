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
Consider the following sentences,<br/>
s1: "**hi how are you**"<br/>
s2: "**hey where are you**"<br/>
To represent the above two sentences in the form of a vector, we dedicate each dimension to a unique word in the dataset. The value in the dimension is 1 if the word is present in the sentence or text and 0 if it is not present.

Sentence | hi | how | are | you | hey | where |
-- | -- | -- | -- | -- | -- | -- |
s1 | 1 | 1 | 1 | 1 | 0 | 0 |
s2 | 0 | 0 | 1 | 1 | 1 | 1 |

This very basic way of representing a sentence is called **Binary Representation**.

## Term Frequency (TF)
Term frequency is more informative than a binary value indicating the presence of a word. In this representation, the value in each dimension is the number of occurences of this word (dimension) in the sentence. For example,<br/>
s3: "**how are you where are you**"

Sentence | hi | how | are | you | hey | where |
-- | -- | -- | -- | -- | -- | -- |
s3 | 0 | 1 | 2 | 2 | 0 | 1

## TF-IDF
A drawback of *TF* metric is, all the words are given equal importance. **Inverse Document Frequency** (IDF) metric assigns importance to word. The intuition behind the definition of IDF is, if a term is present in more documents, then it is less important. For example, the word *a* is present in almost all the documents, so it doesn’t give any **unique** information about this particular document, so it should be given less importance.

![](/images/2020-02-05/tf_idf.png "TF-IDF Score")


## Implementation
### Packages Needed
1. Numpy
2. Pandas
3. Scikit-Learn

### [Download dataset](https://www.kaggle.com/mousehead/songlyrics)

### Import Packages

```python
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

### Read Dataset

```python
df = pd.read_csv('songdata.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>song</th>
      <th>link</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ABBA</td>
      <td>Ahe's My Kind Of Girl</td>
      <td>/a/abba/ahes+my+kind+of+girl_20598417.html</td>
      <td>Look at her face, it's a wonderful face  \nAnd...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ABBA</td>
      <td>Andante, Andante</td>
      <td>/a/abba/andante+andante_20002708.html</td>
      <td>Take it easy with me, please  \nTouch me gentl...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ABBA</td>
      <td>As Good As New</td>
      <td>/a/abba/as+good+as+new_20003033.html</td>
      <td>I'll never know why I had to go  \nWhy I had t...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ABBA</td>
      <td>Bang</td>
      <td>/a/abba/bang_20598415.html</td>
      <td>Making somebody happy is a question of give an...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABBA</td>
      <td>Bang-A-Boomerang</td>
      <td>/a/abba/bang+a+boomerang_20002668.html</td>
      <td>Making somebody happy is a question of give an...</td>
    </tr>
  </tbody>
</table>
</div>



### TF-IDF Search Engine

```python
# Get tf-idf matrix using fit_transform function
vectorizer = TfidfVectorizer()
```

```python
%%time
X = vectorizer.fit_transform(df['text']) # Store tf-idf representations of all docs
```

    CPU times: user 12.6 s, sys: 220 ms, total: 12.9 s
    Wall time: 13.2 s


```python
print(X.shape) # (Number of songs, Number of unique words)
```

    (57650, 82385)


### Query Processing

```python
query = "Take it easy with me, please"
```

```python
%%time
query_vec = vectorizer.transform([query]) # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
results = cosine_similarity(X,query_vec).reshape((-1,)) # Op -- (n_docs,1) -- Cosine Sim with each doc
```

    CPU times: user 72 ms, sys: 7.63 ms, total: 79.7 ms
    Wall time: 79.5 ms


### Print Results

```python
# Print Top 10 results
for i in results.argsort()[-10:][::-1]:
    print(df.iloc[i,0],"--",df.iloc[i,1])
```

    Guns N' Roses -- It's So Easy
    Linda Ronstadt -- It's So Easy (To Fall In Love)
    Kris Kristofferson -- Easy, Come On
    Lorde -- Easy
    Kiss -- Easy As It Seems
    Ne-Yo -- Make It Easy
    Rolling Stones -- It's Not Easy
    Frank Zappa -- Easy Meat
    Billy Joel -- Easy Money
    Stevie Wonder -- Please, Please, Please

## Next Steps
The full code is available in a [Jupyter Notebook](https://github.com/asvskartheek/Text-Retrieval/blob/master/TF-IDF%20Search%20Engine%20(SKLEARN).ipynb) format.
Star the [repo](https://github.com/asvskartheek/Text-Retrieval) and Follow me on [GitHub](https://github.com/asvskartheek). If you find any problems with the code present or have suggestions for improvement, add them into Pull Requests.

## Image References
1. [VSM](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/)
2. [TF-IDF](https://pathmind.com/wiki/bagofwords-tf-idf)