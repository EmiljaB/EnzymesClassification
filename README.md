# Classification of enzymes using Machine Learning and Deep Learning 
The goal of this task is to accurately classify enzymes. We aim to train and test different
algorithms in order to determine which performs better with high complexity of data.
There are six types of enzymes in our dataset: : Oxidoreductases, Transferases, Hydrolases, Lyases, Isomerases, Ligases.

## Dataset
The dataset used for this task is privately owned so it won't be shared in this repository.
There are reoughly 300 instances and 40 features. The dataset is well-balanced.
The data presented high complexity, even though its values were scaled from 0 to 1. 

## Models and Results
The complexity of the data confused the models, so we applied Feature Selection before training and testing the models.
Traditional models: SVM, KNN, Decision Trees, Random Forest, Naive Bayes, Logistic Regression.
Complex models: MLP trained with Backpropagation, MLP with ADELINE, MLP with minibatch, CNN, FCNN, RNN.

Most of the models, especially traditional ones did not perform well, indicating that high complexity of the data inevitably
requires complex models. From all models, FCNN had the highest results.
A Results file is also provided for all algorithms.

## Isemorases illustration
![](https://proteopedia.org/wiki/images/e/e5/D-xylose_isomerase.gif)
