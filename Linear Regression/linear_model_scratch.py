import numpy as np

# load library
import numpy as np
import matplotlib.pyplot as plt

#create data
np.random.seed(0)
x = np.random.rand(100,1)
y = 2 + 3 * x + np.random.rand(100,1)

#plot the data
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

class LinearRegression:
    def __init__(self, alfa=0.05, iterations=1000):
        self.alfa = alfa
        self.iterations = iterations

    def fit(self, x, y):
        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.alfa/m)*gradient_vector
            cost = np.sum((residuals **2))/(2*m)
            self.cost_.append(cost)

        return self

    def predict(self, x):
        return np.dot(x, self.w_)


model = LinearRegression()
model.fit(x, y)
result = model.predict(x)
print(result)