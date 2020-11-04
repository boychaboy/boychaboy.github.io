---
layout: post
title:  "Different Approches to measure bias in Word Embeddings"
subtitle: "Bias Guys - 1"
type: "#NLP #BIAS #FAIRNESS"
blog: true
text: true
author: "boychaboy"
post-header: true
header-img: "img/header.png"
order: 2
---
# Different approches to measure bias in word embeddings

![header](../img/header.png)

It is ground truth that we should figure out ways to detect and mitigate "unintended" bias in machine learning. I don't want to waste my energy here to argue about this idea(if your curious, check out this [paper](https://arxiv.org/abs/2005.14050)). Instead, I want to introduce you several different approaches to **measures such biases in word embeddings**. 

In general, there are two types of word embeddings : "Uncontextualized" and "Contextualized". Example of uncontextualized word embeddings are **Word2Vec** and **GloVe(**Since both are famous and well-known, I won't go over them here). First, I will go over approches to measure bias in uncontextualized word embeddings in binary and multi-class settings each. Second, I will introduce approches to measure bias in contextualiezed word embeddings such as ELMo or BERT.

## 1. Uncontextualized Word Embeddings

### Binary Class

Man is to computer programmer as woman is to homemaker : Debiasing Word Embeddings

![1](../img/1.png)

### Multi-class

Black is to Criminal as Caucasian is to Police:
Detecting and Removing Multiclass Bias in Word Embeddings

## 2. Contextualized Word Embeddings

### ELMo

Gender Bias in Contextualized Word Embeddings.

### BERT

Measuring Bias in Contexturalized Word Representations
