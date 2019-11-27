import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.model_selection import KFold
from scipy.stats import multivariate_normal

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
mu = numpy.array([[-2.0,-2.5,-2.5],[2.5,2.5,2.5],[2.5,2.5,-2.5],[-2.5,-2.5,2.5]])
classprior = numpy.array([0.1,0.2,0.3,0.4])
covEvectors = numpy.zeros([4,3,3])

covEvectors[0] = [[1.9910,-0.6121,-0.7598],
                  [-0.6121,2.2175,0.0505],
                  [-0.7598,0.0505,1.7916]]
covEvectors[1] = [[3.8673,-2.7243,-0.1799],
                  [-2.7243,4.2679,-1.2215],
                  [-0.1799,-1.2215,3.8648]]
covEvectors[2] = [[1.5162,1.2463,-0.0699],
                  [1.2463,9.8501,-4.0840],
                  [-0.0699,-4.0840,3.6337]]
covEvectors[3] = [[3.4376,-2.7815,1.4483],
                  [-2.7815,8.4165,-6.3880],
                  [1.4483,-6.3880,7.1459]]


xTrain = numpy.zeros([Ntrain,3])
t = numpy.random.rand(Ntrain)
L = numpy.zeros([Ntrain])
for i in range(Ntrain):
    if (t[i]<classprior[0]):
        L[i] = 0
        xTrain[i] = numpy.random.multivariate_normal(mu[0],covEvectors[0])
    elif (t[i]>1-classprior[3]):
        L[i] = 3
        xTrain[i] = numpy.random.multivariate_normal(mu[3],covEvectors[3])
    elif (t[i]>classprior[0] and t[i]<classprior[0]+classprior[1]):
        L[i] = 1
        xTrain[i] = numpy.random.multivariate_normal(mu[1],covEvectors[1])
    else:
        L[i] = 2
        xTrain[i] = numpy.random.multivariate_normal(mu[2],covEvectors[2])

fig = plt.figure()
ax = Axes3D(fig)
for i in range(Ntrain):
    if (L[i]==0):
        ax.scatter(xTrain[i,0],xTrain[i,1],xTrain[i,2],color = 'red',label = 'c1')
    elif(L[i]==1):
        ax.scatter(xTrain[i, 0], xTrain[i, 1], xTrain[i, 2], color='blue', label='c2')
    elif(L[i]==2):
        ax.scatter(xTrain[i, 0], xTrain[i, 1], xTrain[i, 2], color='green', label='c3')
    else:
        ax.scatter(xTrain[i, 0], xTrain[i, 1], xTrain[i, 2], color='black', label='c4')

ax.set_xlabel('x0')  # xlabel 方法指定 x 轴显示的名字
ax.set_ylabel('x1')  # ylabel 方法指定 y 轴显示的名字
ax.set_zlabel('x2')
plt.title('traindata')
plt.savefig('data'+str(Ntrain)+'.png')
plt.show()




labels = keras.utils.to_categorical(L, 4)
cnt = 0
score = numpy.zeros([10,2])
ind = 0
Kf = KFold(n_splits=10,shuffle=False)
point = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
avgacc = numpy.zeros([15])
for j in range(15):
    ind=0
    for trainind,testind in Kf.split(xTrain):
        train,test = xTrain[trainind],xTrain[testind]
        Ltrain,Ltest = labels[trainind],labels[testind]
        model = Sequential()
        model.add(Dense(point[j], input_shape=(3,), kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Activation('relu'))
        model.add(Dense(4,activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(lr=0.001),metrics=['accuracy'])
        ori_his = model.fit(train, Ltrain, batch_size=10, epochs=20, verbose=1)
        score[ind] = model.evaluate(test,Ltest, verbose=0)
        ind+=1
        cnt +=1
        print(cnt)
    avgacc[j] = numpy.sum(score.T[1])/10
print(avgacc)

fig3 = plt.figure()
plt.xlabel('perceptrons')
plt.ylabel('acc')
plt.title('model selection')
plt.bar(numpy.arange(15),avgacc)
plt.savefig('model'+str(Ntrain)+'.png')
plt.show()

best = FindIndex(avgacc,max(avgacc))
modelb = Sequential()
modelb.add(Dense(point[best], input_shape=(3,), kernel_initializer='random_uniform', bias_initializer='zeros'))
modelb.add(Activation('relu'))
modelb.add(Dense(4,activation='softmax'))
modelb.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(lr=0.001),metrics=['accuracy'])
ori_hisb = modelb.fit(xTrain, labels, batch_size=10, epochs=20, verbose=1)

Ntest = 10000
xTest = numpy.zeros([Ntest,3])
numpy.random.seed(14)
tt = numpy.random.rand(Ntest)
Lt = numpy.zeros([Ntest])

for i in range(Ntest):
    if (tt[i]<classprior[0]):
        Lt[i] = 0
        xTest[i] = numpy.random.multivariate_normal(mu[0],covEvectors[0])
    elif (tt[i]>1-classprior[3]):
        Lt[i] = 3
        xTest[i] = numpy.random.multivariate_normal(mu[3],covEvectors[3])
    elif (tt[i]>classprior[0] and tt[i]<classprior[0]+classprior[1]):
        Lt[i] = 1
        xTest[i] = numpy.random.multivariate_normal(mu[1],covEvectors[1])
    else:
        Lt[i] = 2
        xTest[i] = numpy.random.multivariate_normal(mu[2],covEvectors[2])


dstt = modelb.predict(xTest)
dst = numpy.zeros([Ntest])
correct = 0
fig4 = plt.figure()
ax4 = Axes3D(fig4)
for i in range(Ntest):
    for j in range(4):
        if (max(dstt[i])==dstt[i,j]):
            dst[i] = j
    if(Lt[i]==dst[i]):
        correct += 1
        ax4.scatter(xTest[i, 0], xTest[i, 1], xTest[i, 2], color='green', label='true')
    else:
        ax4.scatter(xTest[i, 0], xTest[i, 1], xTest[i, 2], color='red', label='false')
ax4.set_xlabel('x0')  # xlabel 方法指定 x 轴显示的名字
ax4.set_ylabel('x1')  # ylabel 方法指定 y 轴显示的名字
ax4.set_zlabel('x2')
plt.title('nnresult:green is true red is false')
plt.savefig('test'+str(Ntrain)+'.png')
plt.show()

acctest = correct/10000
print('acctest is :')
print(acctest)


fig5 = plt.figure()
ax5 = Axes3D(fig5)
for i in range(Ntest):
    if (Lt[i]==0):
        ax5.scatter(xTest[i,0],xTest[i,1],xTest[i,2],color = 'red',label = 'c1')
    elif(Lt[i]==1):
        ax5.scatter(xTest[i,0],xTest[i,1],xTest[i,2], color='blue', label='c2')
    elif(Lt[i]==2):
        ax5.scatter(xTest[i,0],xTest[i,1],xTest[i,2], color='green', label='c3')
    else:
        ax5.scatter(xTest[i,0],xTest[i,1],xTest[i,2], color='black', label='c4')

ax5.set_xlabel('x0')  # xlabel 方法指定 x 轴显示的名字
ax5.set_ylabel('x1')  # ylabel 方法指定 y 轴显示的名字
ax5.set_zlabel('x2')
plt.title('testdata')
plt.savefig('testdata'+str(Ntrain)+'.png')
plt.show()

dMAP = numpy.zeros([10000])
p = numpy.zeros([4])
for i in range(10000):

    var = multivariate_normal(mean=mu[0], cov=covEvectors[0])
    p[0]=var.pdf(xTest[i])*0.1
    var = multivariate_normal(mean=mu[1], cov=covEvectors[1])
    p[1] = var.pdf(xTest[i])*0.2
    var = multivariate_normal(mean=mu[2], cov=covEvectors[2])
    p[2] = var.pdf(xTest[i])*0.3
    var = multivariate_normal(mean=mu[3], cov=covEvectors[3])
    p[3] = var.pdf(xTest[i])*0.4
    if (max(p)==p[1]):
        dMAP[i]=1
    elif(max(p)==p[0]):
        dMAP[i]=0
    elif (max(p) == p[2]):
        dMAP[i] = 2
    else:
        dMAP[i] = 3
correct = 0
fig2 = plt.figure()
ax2 = Axes3D(fig2)
for i in range(10000):
    if(Lt[i]==dMAP[i]):
        correct += 1
        ax2.scatter(xTest[i, 0], xTest[i, 1], xTest[i, 2], color='green', label='true')
    else:
        ax2.scatter(xTest[i, 0], xTest[i, 1], xTest[i, 2], color='red', label='false')
ax2.set_xlabel('x0')  # xlabel 方法指定 x 轴显示的名字
ax2.set_ylabel('x1')  # ylabel 方法指定 y 轴显示的名字
ax2.set_zlabel('x2')
plt.title('mapresult:green is true red is false')
plt.savefig('map.png')
plt.show()

accmap = correct/10000
print('accmap is :')
print(accmap)