# Simple MNIST classification with SVM and CNN

This project is for educationsl purposes only. The goal is to implement classification models in tensorflow.

The task in this project is to perform binary classification. The dataset used for the classification is MNIST. Since MNIST has 10 different target labels, the classification task is to distinguish between odd and even numbers.

The data is imported using `tensorflow` api.

## Linear SVM classifier

`SVMClassifier.py` contains a linear SVM classifier. Classifier accepts any input shape with more than 1 dimensions, and flattens it before training and prediction. It is able to automatically detect two distinct labels and map it to the format neccessary for SVM (SVM requires labels to be 1 or -1).

 The total number of parameters for the classifier is `28*28+1`. It achieves F-score of 0.89.

While training with learning rate of 0.02, 20 epochs and minibatch size of 256, one can obtain similar output

```
Epoch 0, loss: 285.35
Epoch 1, loss: 263.88
Epoch 2, loss: 206.13
Epoch 3, loss: 188.80
Epoch 4, loss: 191.40
Epoch 5, loss: 199.94
Epoch 6, loss: 229.16
Epoch 7, loss: 205.62
Epoch 8, loss: 195.79
Epoch 9, loss: 217.82
Epoch 10, loss: 236.83
Epoch 11, loss: 211.23
Epoch 12, loss: 218.62
Epoch 13, loss: 193.97
Epoch 14, loss: 188.65
Epoch 15, loss: 190.18
Epoch 16, loss: 203.15
Epoch 17, loss: 220.53
Epoch 18, loss: 304.96
Epoch 19, loss: 206.21
Testing score f1: 0.89
```



## CNN classifier

The classifier in `CNNBinaryClassifier.py` reuses the class infrastructure of SVM classifier. The model is not as flexible and MNIST shapes are hard-coded into the method `assemble_graph`. The total number of parameters is less than for SVM, i.e. 742, but the resulting F-score is higher than for linear SVM.

```
Epoch 0, loss: 0.62
Epoch 1, loss: 0.40
Epoch 2, loss: 0.30
Epoch 3, loss: 0.20
Epoch 4, loss: 0.14
Epoch 5, loss: 0.11
Epoch 6, loss: 0.10
Epoch 7, loss: 0.10
Epoch 8, loss: 0.08
Epoch 9, loss: 0.08
Testing score f1: 0.96
```

