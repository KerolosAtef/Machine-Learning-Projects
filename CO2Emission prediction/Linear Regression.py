import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("FuelConsumptionCo2.csv")

# take a look at the dataset
# print(df.head(2))

# summarize the data
# df.describe()

# cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# # cdf.head(9)
#
# #
# # viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
# # viz.hist()
# # plt.show()
# #
# #
# # plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
# # plt.xlabel("FUELCONSUMPTION_COMB")
# # plt.ylabel("Emission")
# # plt.show()
#
cdf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','CO2EMISSIONS']]
msk = np.random.rand(len(df)) < 0.8
# print(cdf)
train = cdf[msk]
test = cdf[~msk]
#
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()
#
# # Simple linear Regression
# regr = linear_model.LinearRegression()
# train_x = np.asanyarray(train[['ENGINESIZE']])
# train_y = np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit (train_x, train_y)
# # The coefficients
# print ('Coefficients: ', regr.coef_)
# print ('Intercept: ',regr.intercept_)
#
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
#
#
# test_x = np.asanyarray(test[['ENGINESIZE']])
# test_y = np.asanyarray(test[['CO2EMISSIONS']])
# test_y_hat = regr.predict(test_x)
#
# print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
# print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
#
# #Multiple Linear Regression
# mulRegr = linear_model.LinearRegression()
# x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
# y = np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit (x, y)
# # The coefficients
# print ('Coefficients: ', mulRegr.coef_,mulRegr.intercept_)
#
all_data = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']]
labels= df[['CO2EMISSIONS']]
x_train, x_test, y_train, y_test =train_test_split(all_data,labels,train_size=0.8,shuffle=True)
model=linear_model.LinearRegression()
model.fit(x_train,y_train)
# print ('Coefficients: ', model.coef_,model.intercept_)
# print(model.score(x_test,y_test))
print('Variance score: %.2f' % model.score(x_test,y_test))
# regr = linear_model.LinearRegression()
# x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
# y = np.asanyarray(train[['CO2EMISSIONS']])
# regr.fit (x, y)
# print ('Coefficients: ', regr.coef_)
# y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
# x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
# y = np.asanyarray(test[['CO2EMISSIONS']])
# print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
# print('Variance score: %.2f' % model.score(x_test,y_test))