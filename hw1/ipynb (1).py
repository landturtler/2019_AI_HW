#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from collections import OrderedDict
import copy


# In[2]:


mnist = np.loadtxt('./mnist.csv', delimiter=',')


# In[3]:


def train_test_split(csv_dataset): # i는 0~100 중의 하나의 수로 train_set의 비율을 나타낸다. ex) 70 => train_set 70% test_set 30%
# numpy matrix는 A[start:end:step] 형태로 표현이 가능하며, 다차원 배열의 원소 중 일부분만 접근하고 싶은 경우 파이썬 슬라이싱(slicing)과 comma(,)로 표현한다.
#먼저 label과 data(픽셀)부분을 나눈다. label은 첫번째 열의 리스트이고, 나머지 열은 data(픽셀)에 해당하므로 위의 기준으로 slicing한다. 이 때, data는 정규화를 위해 256으로 나눈다.

#그 후, 트레이닝 부분과 test부분으로 나눈다.  mnist.csv는 총 10000개의 행을 가지고 있고 0~7999는 training, 8000~9999는 test용으로 사용한다. 이 둘을 나누기 위해 위에서 구한 data와 label matrix를 8000번째 행을 기준으로 나눠 각각 저장한다.

    label = csv_dataset[:, :1]
    data = csv_dataset[:, 1:] /256
    
    train_X = data[:8000, :]
    train_T = label[:8000, :]
    test_X = data[8000: , ]
    test_T = label[8000:, ]
    
    return train_X, train_T, test_X, test_T


# In[4]:


def one_hot_encoding(T): # T is data의 label
# T는 2차원 배열로, 행은 input data의 개수이고 열은 label인 (8000,1)의 형태를 가진다.
# one hot label은 data 개수만큼의 행을 가지고 class개수만큼의 열을 가진 2차원 array 형태로 표현한다. 행의 개수는 T.shape[0], 열의 개수는 10인 matrix이고 처음엔 모든 원소 값을 0으로 초기화한다.
# 그 후 i번째 행에서 label 열만 0을 1로 바꾼다. label 열은  int(T[i])의 형태로 T의 i번째 행의값을 정수로 변환하여 접근한다.
    data = np.zeros((T.shape[0],10))
    for i in range (T.shape[0]):
        data[i][int(T[i])]=1
        i += 1
    return data #one_hot_label


# In[5]:


def Softmax(ScoreMatrix): # 제공.

    if ScoreMatrix.ndim == 2:
        temp = ScoreMatrix
        temp = temp - np.max(temp, axis=1, keepdims=True)
        y_predict = np.exp(temp) / np.sum(np.exp(temp), axis=1, keepdims=True)
        return y_predict
    temp = ScoreMatrix - np.max(ScoreMatrix, axis=0)
    expX = np.exp(temp)
    y_predict = expX / np.sum(expX)
    return y_predict


# In[6]:


def setParam_He(neuronlist):
    
    np.random.seed(1) # seed값 고정을 통해 input이 같으면 언제나 같은 Weight와 bias를 출력하기 위한 함수
# neuronlist의 입력은 각 레이어의 뉴런 개수,[input Layer neuron, hidden layer1 neuron, hidden layer2 neuron, output layer neuron]으로 들어온다.
# W1, b1은 input layer를 받아 forward함수를 사용할 때 쓰이는 weight parameter와 bias 값이며, W2, b2는 hidden layer1을 받아 forward 함수를 사용할 때 쓰이는 weight parameter와 bias값, 마지막으로 W3,b3은 hidden layer2를 받을 때 사용하는 값이다.
# He 방식의 초기화는  W = np.random.randn(fan_in,fan_out) / np.sqrt(fan_in/2) 이므로 W의 fan_in에는 각각의 input, fan_out에는 각각의 output을 대입한다.
#bias의 형태는 X*W와 같은 형태이어야 한다. 즉, b1은 X*W1와 같은 형태이며 이는 neuronlist[1]과 같다. 이와 마찬가지로 b2, b3의 형태를 구하면 된다.  
    W1 = np.random.randn(neuronlist[0], neuronlist[1]) / np.sqrt(neuronlist[0]/2) 
    W2 = np.random.randn(neuronlist[1], neuronlist[2]) / np.sqrt(neuronlist[1]/2)
    W3 = np.random.randn(neuronlist[2], neuronlist[3]) / np.sqrt(neuronlist[2]/2)
    b1 = np.zeros(neuronlist[1])
    b2 = np.zeros(neuronlist[2])
    b3 = np.zeros(neuronlist[3])
    
    return W1, W2, W3, b1, b2, b3


# In[7]:


class linearLayer:
    def __init__(self, W, b):
        #backward에 필요한 X, W, b 값 저장 + dW, db값 받아오기
        
        self.X = None
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
        
        
    def forward(self, x):
        self.X = x
        #내적연산을 통한 Z값 계산
        Z = np.dot(self.X, self.W) + self.b 
        return Z
    #linear score function f(X,W) = X*W + b 이다. np.dot함수를 이용하여 X와 W행렬을 곱하고 마지막으로 bias값을 더하여 score값을 구하였다. 
   
    def backward(self, dZ):
        #백워드 함수
        #backward 함수를 이용하여 최종적으로는 input인 x의 변화량에 따른 output Z의 값 변화량인 dx( dZ/dx)를 구하려 한다. parameter로 dZ가 주어지므로 dx = dZ*W.T 이고 dx와 더불어 dW값과 db값을 update한다.
        #참고로, dW와 db는 각각 dZ/dW, dZ/db를 의미하는 것이다. 
        dx = np.dot(dZ,self.W.T)
        self.dW = np.dot(self.X.T,dZ )
        self.db = np.sum(dZ,axis = 0 )
        
        return dx


# In[8]:


class SiLU:
    def __init__(self):
        self.Z = None # 백워드 시 사용할 로컬 변수
        self.sig = None #백워드 시 사용할 로컬 변수
    
    def forward(self, Z):
        #수식에 따른 forward 함수 작성
        # linear function의 결과값인 Z에 activation function인 SiLU함수를 사용한다. SiLU함수는 x∗sigmoid(x)이다. 즉, 리턴값인 Activation은 앞에서 언급한 식의 x에 Z를 대입한 값이다.
        sig = 1 / (1 + np.exp(-Z))
        Activation = Z * sig
        self.Z = Z
        self.sig = sig
        return Activation
    
    def backward(self, dActivation):
        #input이 Z이고, activation 함수에 의해 만들어진 결과값이 out이라 하면, backward의 최종 결과값은 input의 변화량에 따른 output의 변화량이다. 즉, dx = dout*(activation함수의 미분값)이 리턴 값이다.
        #SILU함수를 f(x)라 하고, x에 대한 sigmoid함수의 결과값을 sig라 하면 f(x) = sig *x, f'(x) = sig + x*(1-sig)*sig이다. 여기에 dActivate(Activate가 forward에서의 out이므로 의미상으로 dout에 해당한다.)를 곱한 값이 backward에서 최종적으로 리턴시키는 값인 input에 대한 미분값 dZ이다.
        dZ = dActivation*(self.sig + self.Z*(1-self.sig)*self.sig)  
        return dZ


# In[9]:


class SoftmaxWithLoss(): # 제공
    
    def __init__(self):
        self.loss = None
        self.softmaxScore = None
        self.label = None
        
    def forward(self, score, one_hot_label):
        
        batch_size = one_hot_label.shape[0]
        self.label = one_hot_label
        self.softmaxScore = Softmax(score)
        self.loss = -np.sum(self.label * np.log(self.softmaxScore + 1e-20)) / batch_size
        
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.label.shape[0]
        dx = (self.softmaxScore - self.label) / batch_size
        
        return dx
                                      


# In[10]:


class dropOut : 
#만약 train_flg가 TRUE인 경우, dropOu의 forward를 진행하며  x와 같은 형상으로 배열을 random하게 생성하고, 그 값이 drop_outㅁratio보다 큰 원소를 TRUE로 설정하여 self.mask에 저장한다. 해당 mask를 기존의 input인 x에 곱한다. 그 결과 dropout_ratio만큼의 뉴런만 존재하게 된다. 만약 drop_out을 실행하고 싶지 않거나 drop_out실행 후 다시 원래의 상태로 돌아가려 한다면 train_flg를 FALSE로 실행하여 1 - self.dropout_ratio(삭제한 비율)를 x에 곱하여 리턴한다.
#backward는 parameter로 out을 미분한 dout에 이미 저장되어있는 self.mask만크의 비율을 곱하여 리턴한다.
#마지막으로 killRate함수를 이용하여 외부에서 ㄴ주어진 dropout_ratio를 self.dropout_ratio에 삽입한다.
    def __init__(self, dropout_ratio = 0) :
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x, train_flg = True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask
    def killRate(self,dropout_ratio) :
        self.dropout_ratio = dropout_ratio


# In[11]:


class ThreeLayerNet :
 #dropout을 수행할 경우를 위해 L1과 SiLU함수 사이, L2와 SiLU함수 사이에 각각 droOut 클래스 객체를 선언하였다. 또한 dropout을 사용하였을 떄의 foward를 진행하여 최종적으로 Loss를 구하는 함수인 forwardWithDropout함수를 선언하였다.

    def __init__(self, paramlist):
        
        W1, W2, W3, b1, b2, b3 = setParam_He(paramlist)
        self.flag = 0
        self.params = {}
        self.params['W1'] = W1
        self.params['W2'] = W2
        self.params['W3'] = W3
        self.params['b1'] = b1
        self.params['b2'] = b2
        self.params['b3'] = b3
        

        self.layers = OrderedDict()
        self.layers['L1'] = linearLayer(self.params['W1'], self.params['b1'])
        self.dropOut1 = dropOut()
        self.layers['SiLU1'] = SiLU()
        self.layers['L2'] = linearLayer(self.params['W2'], self.params['b2'])
        self.dropOut2 = dropOut()
        self.layers['SiLU2'] = SiLU()
        self.layers['L3'] = linearLayer(self.params['W3'], self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()
        
    def scoreFunction(self, x):      
        for layer in self.layers.values():
            # 한 줄이 best 
            #과제2 document를 참고하면 각 layer에서 forward함수를 호출하여 마지막 레이어 전까지의 값을 구할 수 있었다.
            #즉, linear layer와 SiLU 레이어 둘 다 forward함수를 호출할 수 있으므로 self.layer.values를 이용해 dictionary의 값만큼  x값에 지속적으로 layer.forward를 실행하면 activation function 직전까지의 score값을 알 수 있다.
            x = layer.forward(x)     
        score = x
        return score
    
    def forwardWithDropout(self, x, label):
    #dropout용 forward를 구하여 최종 loss값을 리턴하는 함수이다. forward 함수를 이용하여 loss를 구해가는 방향인 것은 맞지만, 히든 레이어에서 일정한 비율로 뉴런을 삭제 후 그 뉴런을 바탕으로 forward를 진행하는 forwardWithDropout함수를 기존의 forward대신 써야 하는 때가 있기 때문에 , 그 부분을 if문으로 처리하였다.
       
        for layer in self.layers:
            x = self.layers[layer].forward(x)
            if layer == 'L1': #layer가 L1인 경우,  다음 for문에서는 SiLU1의 forward가 호출될 것이다.그 전에 dropout1의 forward를 진행해야 한다. 
                x = self.dropOut1.forward(x)
            if layer == 'L2': #layer가 L2인 경우, 다음 for문에서는 SiLU2의 forward가 호출될 것이다. 그 전에 dropout2의 forward를 진행한다. 
                x = self.dropOut2.forward(x)

        return self.lastLayer.forward(x, label)  
    
    def killRate(self, kill_n_h1, kill_n_h2):
        self.flag = 1
        self.dropOut1.killRate(kill_n_h1)
        self.dropOut2.killRate(kill_n_h2)
        
    def forward(self, x, label):
        if(self.flag == 1):
            return self.forwardWithDropout(x,label)
        else:
            score = self.scoreFunction(x)
            return self.lastLayer.forward(score, label)
    
    def accuracy(self, x, label):
        
        score = self.scoreFunction(x)
        score_argmax = np.argmax(score, axis=1)
        
        if label.ndim != 1 : #label이 one_hot_encoding 된 데이터면 if문을 
            label_argmax = np.argmax(label, axis = 1)
            
        accuracy  = np.sum(score_argmax==label_argmax) / int(x.shape[0])
        
        return accuracy
 
    def backpropagation(self, x, label):
        
        #백워드 함수 작성 스코어펑션을 참고하세요 lastlayer는 ordered dictionary가 아니기 때문에 가장 먼저 호출 후 그 값을 변수로 저장한다.
        #그 이후 orderd diction의 파이썬 문법인 reversed함수를 구현하여 forward 순서의 반대 순서로 접근이 가능하다. scorefunction와 같은 방식으로 forward 대신 backward를 실행하면 그대로 backward 연산하면 backpropagation을 구현할 수있다.
        #backward 함수를 실행하면 dx값 뿐만 아니라 dW와 db의 값도 얻을 수 있으므로 해당 값들을 최종적으로 grad배열의 dW,db에 update시킨다.
        backx = self.lastLayer.backward()
        for layer in reversed(self.layers.values()):
            backx = layer.backward(backx)
        
        grads = {}
        grads['W1'] = self.layers['L1'].dW
        grads['b1'] = self.layers['L1'].db
        grads['W2'] = self.layers['L2'].dW
        grads['b2'] = self.layers['L2'].db
        grads['W3'] = self.layers['L3'].dW
        grads['b3'] = self.layers['L3'].db
        
        return grads
    
    def gradientdescent(self, grads, learning_rate):
        
        self.params['W1'] -= learning_rate*grads['W1']
        self.params['W2'] -= learning_rate*grads['W2']
        self.params['W3'] -= learning_rate*grads['W3']
        self.params['b1'] -= learning_rate*grads['b1']
        self.params['b2'] -= learning_rate*grads['b2']
        self.params['b3'] -= learning_rate*grads['b3']
        


# In[12]:


def batchOptimization(dataset, ThreeLayerNet, learning_rate, epoch=1000):
  #epoch가 10번째일때마다 accuracy를 체크한다.위 함수에서는 아직 train_acc_list와 test_acc_list,Loss_list가 선언이 되어있지 않았으므로 선언을 해야 한다. 위의 세 리스트를 선언 후 accuracy에 저장한다. 빈 리스트의 선언은 '리스트명' = []로 선언하고 원소 추가는 append를 사용한다.     
    train_acc_list = [] 
    test_acc_list = []
    Loss_list = [] 
    
    for i in range(epoch+1):

#ThreeLayerNet의 forward함수는  마지막 layer 전까지 score를 구한 후 activation function을 통해 입력으로 들어온 data에 대한 Loss값을 구한다. 이를 이용하여 data_train과 data_test의 Loss를 구한다. 

#이제 loss값을 backpropergation을 하여 loss값이 줄어드는 방향으로 W와 b값을 최적화 해야 한다. backpropagation함수 역시threeLayerNet에 구현되어있으므로 해당 함수를 불러올 수 있다. backpropagation함수의 결과로 세 레이어 각각의 dW와 db값을 저장한다. 그런데 이 값을 그대로 update하는 것이 아니라 learning rate만큼 곱한 정도만큼을 W와 b에 update해야 하므로 gradientdescent 함수를 불러 learning rate를 곱한 정도만큼 값을 update한다. update된 값들은 self.params에 저장이 되며 이 과정까지가 한 epoch당 optimization하는 것이 된다. 이를 epoch만큼 실행하며, 10번마다 train_acc_list, test_acc_list,Loss_list에 저장 후 출력한다. 최종 리턴값은 언급한 list에 Threelayernet을 포함한다.

#한편, dictionary 객체는 순서가 없기 때문에 key값으로 접근할 수 있으며,  key값 접근은 ' ' 형식이므로 dataset의 train_data와 one_hot_train(label을 one-hot-encoding형식으로 바꾼 것)를 dataset['train_data'], dataset['one_hot_train']형식으로 접근한다. 다른 data들 역시 위의 방식으로 접근한다. 따라서 함수를 구현하면 다음과 같은 방식으로 구현이 가능하다. 
        Loss = ThreeLayerNet.forward(dataset['train_X'], dataset['one_hot_train'])
        back = ThreeLayerNet.backpropagation(dataset['train_X'], dataset['one_hot_train'])
        ThreeLayerNet.gradientdescent(back, learning_rate)
            
        if (i % 10 == 0) and ((i//10) < 3 or ( i//10) >97):
            train_acc = ThreeLayerNet.accuracy(dataset['train_X'], dataset['one_hot_train'])
            test_acc = ThreeLayerNet.accuracy(dataset['test_X'], dataset['one_hot_test'])
            print(i, '\t번째 Loss = ', Loss)
            print(i, '\t번째 Train_Accuracy : ', train_acc)
            print(i, '\t번째 Test_Accuracy : ', test_acc)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            Loss_list.append(Loss)        
   
    return ThreeLayerNet, train_acc_list, test_acc_list, Loss_list


# In[13]:


def minibatch_Optimization(dataset, ThreeLayerNet, learning_rate, epoch=100, batch_size=100):    
    
    np.random.seed(5)
    for i in range(epoch+1):
        # 코드 작성
        #train_X와 one_hot_train을 np.random.shuffle을 사용해서 섞는다. 대신 한 쌍 단위로 섞어 한 쌍의 단위처럼 셔플이 되어야 한다. 
        #마찬가지로 train_acc_list, test_acc_list,Loss_list가 선언되어있지 않으므로 선언을 한다.
        train_X = dataset['train_X']
        train_T = dataset['one_hot_train']
       # train_acc_list = []
        #test_acc_list = []
        #Loss_list = []  
       # set = list(zip(train_X,train_T))
        #np.random.shuffle(set)
        #train_X,train_T = zip(*set)
        
        #shuffle된 data를 batch단위로 나눠서 forward와 backpropagation, gradientdescent를 한다. 
        #numpy.arange(Start,stop,step)함수를 사용하여 data를 batch_size만큼의 사이즈로 슬라이싱할 것이다. train_X의 행의 개수가 전체 data를 의미 하므로, 0부터 train_X.shape[0]을 이용한다. 자르는 단위는 batch_size(100)만큼 나눈다. 이 결과로 miniSize에는 minibatch로 나누어질 수 있는 배열의 개수가 나온다.
        #for문을 minibatch만큼 돌려서 batch_size만큼 train_X와 train_T 배열을 슬라이싱 한 후 각각 forward와 backward, gradient descent를 시킨다. 
#int형 랜덤변수 seed값을 받아서 해당 seed값으로 data와 label에 동일하게 셔플을 진행시킨다. 즉 setseed라는 값으로 seed값을 변수로 저장시키고 그 값으로 shuffle을 진행해야 train과 label의 순서가 바뀌지 않는다. 
        miniSize = np.arange(0,train_X.shape[0], batch_size)
        setseed = np.random.randint(0,10000)
        np.random.seed(setseed)
        train_X = np.random.shuffle(train_X)
        np.random.seed(setseed)
        train_T = np.random.shuffle(train_T)
        
        for j in miniSize:
            mini_X = dataset['train_X'][j: j + batch_size,:]
            mini_T = dataset['one_hot_train'][j: j + batch_size,:]
    
            Loss = ThreeLayerNet.forward(mini_X, mini_T)
            back = ThreeLayerNet.backpropagation(mini_X, mini_T)
            ThreeLayerNet.gradientdescent(back, learning_rate)
        
        if (i % 10 == 0) and ((i//10) < 3 or ( i//10) >8):
            train_acc = ThreeLayerNet.accuracy(dataset['train_X'], dataset['one_hot_train'])
            test_acc = ThreeLayerNet.accuracy(dataset['test_X'], dataset['one_hot_test'])
            print(i, '\t번째 Loss = ', Loss)
            print(i, '\t번째 Train_Accuracy : ', train_acc)
            print(i, '\t번째 Test_Accuracy : ', test_acc)
            #train_acc_list.append(train_acc)
            #test_acc_list.append(test_acc)
            #Loss_list.append(Loss)  

    return ThreeLayerNet #, train_acc_list, test_acc_list, Loss_list


# In[14]:


def dropout_use_Optimizer(dataset, ThreeLayerNet, learning_rate, epoch, kill_n_h1 = 0.25, kill_n_h2 = 0.15):
#ThreeLayerNet.killRate함수를 이용하면 해당 클래스 내의 self.flag 값이 1로 setting이 된다. 이는 ThreeLayerNet에서 forward연산 진행 시 flag=1이므로 dropout을 사용하는 if문의 함수가 실행되어 dropout 방식을 사용하게 된다. 
#그렇기 때문에 현재 함수에서는 killRate를 설정하는 것 외에는 기존의 batch optimizer방식과 동일하다.

    ThreeLayerNet.killRate(kill_n_h1, kill_n_h2) #threeLayerNet에 kill rate를 세팅한다. 
    train_acc_list = []
    test_acc_list = []
    Loss_list = []     
    for i in range(epoch+1):
        #코드 작성
        Loss = ThreeLayerNet.forward(dataset['train_X'], dataset['one_hot_train'])
        back = ThreeLayerNet.backpropagation(dataset['train_X'], dataset['one_hot_train'])
        ThreeLayerNet.gradientdescent(back, learning_rate)
    
    
        if (i % 10 == 0) and ((i//10) < 3 or ( i//10) >97):
            train_acc = ThreeLayerNet.accuracy(dataset['train_X'], dataset['one_hot_train'])
            test_acc = ThreeLayerNet.accuracy(dataset['test_X'], dataset['one_hot_test'])
            print(i, '\t번째 Loss = ', Loss)
            print(i, '\t번째 Train_Accuracy : ', train_acc)
            print(i, '\t번째 Test_Accuracy : ', test_acc)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            Loss_list.append(Loss)  
    return ThreeLayerNet, train_acc_list, test_acc_list, Loss_list


# In[15]:


#과제 채점을 위한 세팅
train_X, train_label, test_X, test_label = train_test_split(mnist)

one_hot_train = one_hot_encoding(train_label)
one_hot_test = one_hot_encoding(test_label)

dataset = {}
dataset['train_X'] = train_X
dataset['test_X'] = test_X
dataset['one_hot_train'] = one_hot_train
dataset['one_hot_test'] = one_hot_test

neournlist = [784, 60, 30, 10]

TNN_batchOptimizer = ThreeLayerNet(neournlist)
TNN_minibatchOptimizer = copy.deepcopy(TNN_batchOptimizer)
TNN_dropout = copy.deepcopy(TNN_minibatchOptimizer)


# In[16]:


#채점은 이 것의 결과값으로 할 예정입니다. 

trained_batch, tb_train_acc_list, tb_test_acc_list, tb_loss_list =  batchOptimization(dataset, TNN_batchOptimizer, 0.1, 1000)
trained_minibatch = minibatch_Optimization(dataset, TNN_minibatchOptimizer, 0.1, epoch=100, batch_size=100)
trained_dropout, td_train_acc_list, td_test_acc_list, td_loss_list = dropout_use_Optimizer(dataset, TNN_dropout, 0.1, 1000, 0.25, 0.15)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




