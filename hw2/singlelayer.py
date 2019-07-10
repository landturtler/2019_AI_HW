import numpy as np


class singleLayer :
    def __init__(self, W, Bias): # 제공. 호출 시 작동하는 생성자
        self.W = W
        self.B = Bias

    def SetParams(self, W_params, Bias_params): # 제공. W와 Bias를 바꾸고 싶을 때 쓰는 함수
        self.W = W_params
        self.B = Bias_params

    def ScoreFunction(self, X): # \Score값 계산 -> 직접작성
        ScoreMatrix =np.dot(X,self.W) + self.B #linear score function f(X,W) = X*W + b 이다. np.dot함수를 이용하여 X와 W행렬을 곱하고 마지막으로 bias값을 더하여 score값을 구하였다. scoreMatrix의 행렬은 [60000, 10]이 된다.
        return ScoreMatrix

    def Softmax(self, ScoreMatrix): # 제공
        if ScoreMatrix.ndim == 2: #scorefunction의 값이 2차원인 경우( 여기서는 항상 2차원이다)
            temp = ScoreMatrix.T #ScoreMatrix 행렬을 전치시킨다. 따라서 temp는 [10, 1]의 matrix가 된다.
            temp = temp - np.max(temp, axis=0)
            y_predict = np.exp(temp) / np.sum(np.exp(temp), axis=0)
            return y_predict.T
        temp = ScoreMatrix - np.max(ScoreMatrix, axis=0)
        expX = np.exp(temp)
        y_predict = expX / np.sum(expX)
        return y_predict #정답 label의 score값을 평균으로 표현, 즉 정답일 확률. 이 값을 -log를 취하면 loss값이 나온다.

    def LossFunction(self, y_predict, Y): #  Loss Function을 구하십시오 -> 직접 작성, Y는 score function의 결과값(진짜 확률)
        delta = 1e-7
        estimated = -np.log(y_predict+delta) #이미지의 label값으로 추정된 확률(softmax결과값)에 -log를 씌운 값, 즉 추정된 정보량을 의미한다. 단, y_predict가 0일때 inf가 나와 결과값이 불안정해 질 수 있으므로 delta를 더하여 inf값이 나오지 못하도록 한다
        size = y_predict.shape[0]  # 이미지들을 학습시킬 때 사용한 score function에 대한 전체 loss는 Li의 평균이다. 따라서 softmax 결과값인 y_predict의 합을 구한 후 전체 input 개수만큼 나누어야 한다. y_predict[60000,10]에서 60000은 data개수, 10은 class개수이므로 배열의 크기를 나타내는 shape함수를 사용하여 shape[0]=60000을 받아온다.
        loss = np.sum(Y*estimated) / size#cross Entropy를 기반으로  실제 확률 * ( - 추정된 정보량) 결과값을 더한 후 data 개수만큼 나누면 전체 loss가 반환된다.
        return loss

    def Forward(self, X, Y): # ScoreFunction과 Softmax, LossFunction를 적절히 활용해 y_predict 와 loss를 리턴시키는 함수. -> 직접 작성
        ScoreMatrix = self.ScoreFunction(X) #input image X(60000,784)에 linear score function을 취한 결과값은 X의 ScoreMatrix(60000,10)가 된다.
        y_predict = self.Softmax(ScoreMatrix) #그리고 해당 ScoreMatrix를 softmax function을 취해 각 이미지에서 정답 label의 score값을 평균으로 표현한 결과가 y_predict(60000,10)이면서 softmax값이다.
        loss = self.LossFunction(y_predict,Y) #y_predict의 정보량을 이용하여 cross-entropy를 이용한 loss function(오차값)을 취한 값이 loss(60000,10)이다.
        return y_predict, loss



    def delta_Loss_Scorefunction(self, y_predict, Y): # 제공.dL/dScoreFunction
        delta_Score = y_predict - Y
        return delta_Score

    def delta_Score_weight(self, delta_Score, X): # 제공. dScoreFunction / dw .
        delta_W = np.dot(X.T, delta_Score) / X[0].shape
        return delta_W

    def delta_Score_bias(self, delta_Score, X): # 제공. dScoreFunction / db .
        delta_B = np.sum(delta_Score) / X[0].shape
        return delta_B

    # delta 함수를 적절히 써서 delta_w, delta_b 를 return 하십시오.
    def BackPropagation(self, X, y_predict, Y):
        delta_Score = self.delta_Loss_Scorefunction(y_predict,Y) #dW와 dB를 구하기 위해서는 먼저 dL/df값이 필요하다. delta_Score는 dL/dScoreFunction을 반환한다.
        delta_W = self.delta_Score_weight(delta_Score,X) #delta_score_weight함수는 delta_score와 X값을 이용하여 dF/dW를 반환하는 함수이다. 위에서 delta_Score값을 구했으므로 해당 값을 매개변수로 넣어 dW를 구한다.
        delta_B = self.delta_Score_bias(delta_Score,X) #위와 마찬가지로 delta_score를 구헀으므로 dF/db를 구할 수 있다.
        return delta_W, delta_B


    # 정확도를 체크하는 Accuracy 제공
    def Accuracy(self, X, Y):
        y_score = self.ScoreFunction(X)
        y_score_argmax = np.argmax(y_score, axis=1)
        if Y.ndim!=1 : Y = np.argmax(Y, axis=1)
        accuracy =100 * np.sum(y_score_argmax == Y) / X.shape[0]
        return accuracy

    # Forward와 BackPropagationAndTraining, Accuracy를 사용하여서 Training을 epoch만큼 시키고, 10번째 트레이닝마다
    # Training Set의 Accuracy 값과 Test Set의 Accuracy를 print 하십시오

    def Optimization(self, X_train, Y_train, X_test, Y_test, learning_rate = 0.01, epoch=100):
        for i in range(epoch):
            y_predict, loss = self.Forward(X_train,Y_train)#트레이닝 할 data set을 통해 해당 이미지의 Li를 구한다.  y_predict는 ScoreMatrix를 softmax function을 취해 각 이미지에서 정답 label의 score값을 평균으로 표현한 결과(Li)이고, loss값은 각 Li의 평균을 구한 것이다.
            delta_W, delta_B = self.BackPropagation(X_train,y_predict,Y_train) #이제 backpropagationd을 사용하여 L값이 h만큼 증가했을 때, W와 B의 변화(혹은 그 반대)를 나타내는 함수를 사용한다. 리턴값은 dF/dW,dF/db이다.
            #-learning rate만큼 곱하여서 gradeint descent를 실행한다. 위에서 구한 delta_W와 delta_B는 gradient를 의미한다. -gradient(벡터)는 f의 값이 가장 가파르게 감소하는 방향을 나타내므로 우리는 W가 최소가 되는 지점을 찾  아야 하기 때문에 일정한 rate만큼 -gradient 방향으로 이동한다면 최적화된 W에 가까워 질 수 있을 것이다.
            self.W -= learning_rate * delta_W #W값과 bias값을 최적화 W에 가까운 값으로 이동시킨다.
            self.B -= learning_rate + delta_B
             #위의 방식으로 트레이닝을 하나의 image당 100번만큼 지속하여 loss값이 최소가 되는 W와 B의 값을 구해간다.

            if i % 10 == 0: #10번째 트레이닝일 경우, Accuracy를 출력한다.
                #3.6 Accuracy 함수 사용
                print(i, "번째 트레이닝")
                print('현재 Loss(Cost)의 값 : ', loss)
                accutraing = self.Accuracy(X_train, Y_train)
                print("Train Set의 Accuracy의 값 : ",accutraing )
                accutest= self.Accuracy(X_test,Y_test)
                print("Test Set의 Accuracy의 값 :", accutest)
