import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import KFold

def FindIndex(source, Destina):
    i = 0;
    for iterating_var in source:
        if(iterating_var == Destina):
            return i;
        elif(iterating_var <= Destina):
            i = i+1;
    return i;

Ntrain = 1000
n = 2
K = 10
mu = numpy.array([[-18,-8],[0,0],[18,8]])
classprior = numpy.array([0.33,0.34,0.33])
covEvectors = numpy.zeros([3,2,2])
covEvectors[0] = [[1/math.sqrt(2),-1/math.sqrt(2)],[1/math.sqrt(2),1/math.sqrt(2)]]
covEvectors[1] = [[1,0],[0,1]]
covEvectors[2] = [[1/math.sqrt(2),-1/math.sqrt(2)],[1/math.sqrt(2),1/math.sqrt(2)]]
covEvalues = numpy.array([[numpy.square(3.2),0],[0,numpy.square(0.6)]])

xTrain = numpy.zeros([Ntrain,2])
t = numpy.random.rand(Ntrain)
for i in range(Ntrain):
    if (t[i]<classprior[0]):
        xTrain[i] = numpy.dot(covEvectors[0],numpy.dot(covEvalues**(1/2),numpy.random.randn(2)))+mu[0]
    elif (t[i]>1-classprior[2]):
        xTrain[i] = numpy.dot(covEvectors[2], numpy.dot(covEvalues ** (1 / 2), numpy.random.randn(2)))+mu[2]
    else:
        xTrain[i] = numpy.dot(covEvectors[1], numpy.dot(covEvalues ** (1 / 2), numpy.random.randn(2))) + mu[1]
fig = plt.figure()
plt.scatter(xTrain.T[0],xTrain.T[1])
plt.title('data')
plt.savefig('2data.png')
plt.show()

Ntest = 10000

xtest = numpy.zeros([Ntest,2])
t = numpy.random.rand(Ntest)
for i in range(Ntest):
    if (t[i]<classprior[0]):
        xtest[i] = numpy.dot(covEvectors[0],numpy.dot(covEvalues**(1/2),numpy.random.randn(2)))+mu[0]
    elif (t[i]>1-classprior[2]):
        xtest[i] = numpy.dot(covEvectors[2], numpy.dot(covEvalues ** (1 / 2), numpy.random.randn(2)))+mu[2]
    else:
        xtest[i] = numpy.dot(covEvectors[1], numpy.dot(covEvalues ** (1 / 2), numpy.random.randn(2))) + mu[1]






# model = Sequential()
#
# model.add(Dense(6, input_shape=(1,),kernel_initializer='random_uniform',bias_initializer='zeros'))
# model.add(Activation('softplus'))
# model.add(Dense(1))
# model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.SGD(lr=0.001))
score = numpy.zeros([10])

Kf = KFold(n_splits=10,shuffle=False)
point = [1,2,3,4,5,6,7,8,9,10,11,12,13]
loss = numpy.zeros([13])
cont = 0
for j in range(13):
    i = 0
    for trainind,testind in Kf.split(xTrain):
        train,test = xTrain[trainind],xTrain[testind]
        model = Sequential()
        model.add(Dense(point[j], input_shape=(1,), kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Activation('softplus'))
        model.add(Dense(1))
        model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.adam(lr=0.001))
        ori_his = model.fit(train.T[0].T, train.T[1].T, batch_size=1, epochs=10, verbose=1)
        score[i] = model.evaluate(test.T[0].T,test.T[1].T, verbose=0)
        i+=1
        cont += 1
        print(cont)
    loss[j] = numpy.sum(score)/10

lossp = numpy.zeros([13])
for j in range(13):
    i = 0
    for trainind,testind in Kf.split(xTrain):
        train,test = xTrain[trainind],xTrain[testind]
        model = Sequential()
        model.add(Dense(point[j], input_shape=(1,), kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Activation('sigmoid'))
        model.add(Dense(1))
        model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.adam(lr=0.001))
        ori_his = model.fit(train.T[0].T, train.T[1].T, batch_size=1, epochs=10, verbose=1)
        score[i] = model.evaluate(test.T[0].T,test.T[1].T, verbose=0)
        i+=1
        cont +=1
        print(cont)
    lossp[j] = numpy.sum(score)/10
print(loss)
print(lossp)
fig2 = plt.figure()
plt.scatter(numpy.arange(13),loss,c="b",marker="x",label='softplus')
plt.scatter(numpy.arange(13),lossp,c="r",label='sigmoid')
plt.xlabel('perceptrons')  # xlabel 方法指定 x 轴显示的名字
plt.ylabel('loss')  # ylabel 方法指定 y 轴显示的名字
plt.title('different models loss')
plt.legend()
plt.savefig('2_1.png')
plt.show()
best = FindIndex(loss,min(loss))
bestp = FindIndex(lossp,min(lossp))
if min(loss) <= min(lossp):
    model1 = Sequential()
    model1.add(Dense(point[best], input_shape=(1,), kernel_initializer='random_uniform', bias_initializer='zeros'))
    model1.add(Activation('softplus'))
    model1.add(Dense(1))
    model1.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.adam(lr=0.001))
    ori_his1 = model1.fit(xTrain.T[0].T, xTrain.T[1].T, batch_size=1, epochs=50, verbose=1)

else:
    model1 = Sequential()
    model1.add(Dense(point[bestp], input_shape=(1,), kernel_initializer='random_uniform', bias_initializer='zeros'))
    model1.add(Activation('sigmoid'))
    model1.add(Dense(1))
    model1.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.adam(lr=0.001))
    ori_his1 = model1.fit(xTrain.T[0].T, xTrain.T[1].T, batch_size=1, epochs=50, verbose=1)


fig2 = plt.figure()
y = model1.predict(xtest.T[0].T)
plt.scatter(xtest.T[0],xtest.T[1],c="b",label = 'true')
plt.scatter(xtest.T[0],y,c="r",label = 'decision')
plt.xlabel('x1')  # xlabel 方法指定 x 轴显示的名字
plt.ylabel('x2')  # ylabel 方法指定 y 轴显示的名字
plt.title('test result')
plt.legend()
plt.savefig('2_2.png')
plt.show()
