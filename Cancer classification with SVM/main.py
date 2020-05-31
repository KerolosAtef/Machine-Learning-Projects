import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("cell_samples.csv")
    # print(df.columns)
    # print(df.dtypes)
    # It looks like the BareNuc column includes some values that are not numerical. We can drop those rows:
    df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
    x = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
            'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']].values.astype(int)
    y = df['Class'].values.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
    # rbf kernel is the Gaussian kernel
    svm_rbf_kernel_model = svm.SVC(C=2, kernel='rbf')
    svm_rbf_kernel_model.fit(x_train, y_train)
    train_acc = svm_rbf_kernel_model.score(x_train, y_train)
    test_acc = svm_rbf_kernel_model.score(x_test, y_test)
    print("RBF model")
    print("train accuracy", train_acc)
    print("test accuracy", test_acc)
    svm_linear_kernel_model = svm.SVC(C=2, kernel='linear')
    svm_linear_kernel_model.fit(x_train, y_train)
    train_acc = svm_linear_kernel_model.score(x_train, y_train)
    test_acc = svm_linear_kernel_model.score(x_test, y_test)
    print("Linear model")
    print("train accuracy", train_acc)
    print("test accuracy", test_acc)
    svm_polynomial_kernel_model = svm.SVC(C=2, kernel='poly',degree=3)
    svm_polynomial_kernel_model.fit(x_train, y_train)
    train_acc = svm_polynomial_kernel_model.score(x_train, y_train)
    test_acc = svm_polynomial_kernel_model.score(x_test, y_test)
    print("Polynomial model")
    print("train accuracy", train_acc)
    print("test accuracy", test_acc)

    svm_sigmoid_kernel_model = svm.SVC(C=2, kernel='sigmoid')
    svm_sigmoid_kernel_model.fit(x_train, y_train)
    train_acc = svm_sigmoid_kernel_model.score(x_train, y_train)
    test_acc = svm_sigmoid_kernel_model.score(x_test, y_test)
    print("Sigmoid model")
    print("train accuracy", train_acc)
    print("test accuracy", test_acc)
