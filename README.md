# 2023 CS429/529 Project 2: Topic Categorization

> *Second project for the CS ML class at UNM (Spring 2023)*
> 
> (Full instructions in [Kaggle](https://www.kaggle.com/competitions/cs429529-project-2-topic-categorization))

## Prerequisites

This program is written entirely on **Python**. The libraries used for this project are indicated in the `requirements.txt` file. These include:
- ipykernel
- ipython
- jupyter
- jupyter-client
- jupyter-core
- jupyterlab-pygments
- numpy
- pandas
- matplotlib
- scikit-learn
- tqdm
- bokeh
- imbalanced-learn 

The data necessary to run the code can be found in [the official 20 Newsgroups site](http://qwone.com/~jason/20Newsgroups) or [Kaggle](https://www.kaggle.com/competitions/cs429529-project-2-topic-categorization/data). Make sure to download it and place it in the same folder (or update the locations in the notebooks themselves) before running the notebooks.

## Execution

The code can be ran directly from the notebooks.

## Description

The following is a project about **Naïve Bayes** and **Logistic Regression**. Its objective is to understand how to implement these models from scratch to understand how they work. A bag of words representation is used.

### Naïve Bayes

Classifies according to the following rule: $$h_{NB}(X) = \arg\!\max_{y \in Y}P(y)\prod_{i=1}^{n}P(X_{i}|y)$$

### Logistic Regression

Classifies following the rule:
$$lnP(D_{Y}| D_{X}, w) = \sum_{j=1}y^{j}(w_{0}+\sum_{i}^{n}w_{i}{x^{j}}_{i}) - ln(1 + exp(w_{0} + \sum_{i}^{n}w_{i}{x^{j}}_{i}))$$

## Results

The results can be found in more detail on the report (`TopicCategorizationCS429.pdf`)