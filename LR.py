
class LinearRegression:

    def __init__(self, learning_rate=0.000005, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.listM1 = []
        self.listM2 = []
        self.listB = []

    def calculateBIM(self, height: float, weight: float):
        return (self.m1*height)+(self.m2*weight)+self.b
    
    def loss(self, X_train, Y_train, Z_train):

        Z_prediction = self.predict(X_train, Y_train)

        self.n = len(X_train)

        m1_gradient = 0
        m2_gradient = 0
        b_gradient = 0
        
        for row in range(len(X_train)):
            temp = Z_prediction[row] - Z_train[row]
            m1_gradient += (2 / self.n) * (X_train[row] * temp)
            m2_gradient += (2 / self.n) * (Y_train[row] * temp)
            b_gradient += (2 / self.n) * (temp)

        self.listM1.append(m1_gradient)
        self.listM2.append(m2_gradient)
        self.listB.append(b_gradient)
        
        self.m1 = self.m1 - self.learning_rate * m1_gradient
        self.m2 = self.m2 - self.learning_rate * m2_gradient
        self.b = self.b - self.learning_rate * b_gradient 

        
        
        return self

    def fit(self, X_train, Y_train, Z_train):
        list_length = len(X_train)
        self.m1 = 1
        self.m2 = 2
        self.b = 0

        for i in range(self.epoch):
            self.loss(X_train, Y_train, Z_train)

        return self
    
    def predict(self, X_test, Y_test):
        prediction = []

        for i in range(len(X_test)):
            prediction.append(self.calculateBIM(X_test[i], Y_test[i]))

        return prediction
    def getLossList(self):
        return self.listM1, self.listM2, self.listB

model = LinearRegression()


