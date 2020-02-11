# Dealing With Ordinal Data
Ordinal data is a categorical, statistical data type where the variables have natural, ordered categories and the distances between the categories is not known. To understand the order within the data and methods of modelling, we'll introduce a new variable.

Let the data that we have which is of ordinal nature be ($m$). The variable introduced be ($n$), this is a free variable i.e, not categorical but continuous and real.

The variable $n$ can be distributed in anyway. But for the discussion purposes, let us assume it to be following a normal distribution. The same reasoning can be applied to any other distributions too.

## Assumption 1: $n$ follows Normal

![](/images/2020-02-11/normal_img.png "Probability Distribution showing categories")

For example, let $m$ be a 5 unit linkert scale variable which can be seen in the above figure. Mapping from $n$ to $m$ can be made using this "**Categorising Function**"
$$
n\geq Thresh_1: m=1 \newline
Thresh_2\leq n < Thresh_1 : m=2 \newline
Thresh_3\leq n < Thresh_2 : m=3 \newline
Thresh_4\leq n < Thresh_3 : m=4 \newline
n < Thresh_4: m = 5
$$

So, to get the value of $m$ we need to get the value of $n$ and the knowledge of thresholds i.e, knowing $Thresh_1$ - $Thresh_4$

A simple data driven approach to get the values of thresolds, is based on the frequentist idea of probability i.e, number of times the value occuring. (In real life, the value of thresholds are fuzzy and depend on the user's lineancy but for this discussion we assume $Thresh_1$ - $Thresh_4$ are global constants.)

## Assumption 2: Threshold values are global constants.

Without loss of generality, we can assume that $n$ to following a standard normal as the constant offset can be modelled using a bias term.

$$
P(m=1) = P(n \geq Thresh_1) \newline
\implies P(m=1) = N(n \geq Thresh_1)
$$

$P(m=1)$ can be calculated from the sample data & using a point estimate for the population. $N$ is a standard normal, so using tables, we can get the appropriate $Thresh_1$ value. Similarly, other threshold values can be calculated.
