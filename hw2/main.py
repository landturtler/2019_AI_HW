
import numpy as np
from mnist import load_mnist
import singlelayer as sn #SingleayerNetwork
from PIL import Image

np.random.seed(1)

def img_show(img): # (784) 혹은 (28, 28)의 mnist 배열을 이미지로 보여주는 작업
   if img.ndim==1: #image가 1차원, 즉 흑백 이미지인 경우
      img = img.reshape(28, 28) #이미지의 크기를 (28*28)의 mnist배열로 바꿈
      img *= 255.0 #이미지가 저장된 픽셀 값(0부터 1사이의 값)을 0~255사이의 값으로 변환하기 위해 255를 곱함
      pil_img = Image.fromarray(np.uint8(img)) #array interface를 내보내는 unit8(img)로부터 이미지 메모리를 생성
      pil_img.show() #해당 이미지 출력


def get_data(): # mnist 데이터 받아오기(one_hot_label로)
   (x_train, y_train), (x_test, y_test) = \
   load_mnist(normalize=True, flatten=True, one_hot_label=True)
   return x_train, y_train, x_test, y_test

def is_number(s):
   try:
      int(s) #문자열을 정수로 변환
      return True
   except ValueError: #오류발생
      return False

def TestSN(input_i, x_train, y_train, x_test, y_test, W, Bias): #test 이미지를 하나 뽑고 singleNN.py의 trainingAndResult를 돌려서 학습전 결과와 학습 후 결과가 어떻게 다른지 확인하는 것
   if is_number(input_i): #input_i를 정수로 변환이 가능하다면 문자열 input_i를 정수로 변환
      i = int(input_i)
      Test = x_train[i]
      label = np.argmax(y_train[i]) #y_train[i]에 해당하는 값들 중 가장 큰 값의 인덱스, 즉 label의 index를 반환
      img_show(Test) #Test이미지 출력
      print("이 이미지의 실제 값 : ", label) #그림의 숫자와 동일
      SN = sn.singleLayer(W, Bias)
      y_predict = SN.ScoreFunction(x_train[i]) #mnist data에 대해 score 계산
      print("이 이미지의 학습 전 이미지의 추론 값 : ", np.argmax(y_predict)) #추론값의 결과가 그림의 숫자와 같을 수도 다를 수도 있음.
      SN.Optimization(x_train, y_train, x_test, y_test) #optimization 실행
      y_predict = SN.ScoreFunction(x_train[i]) #optimization 실행 후의 추론값을 다시 구함
      print("학습이 완료되었습니다 \n이미지의 학습 후 추론 값: ", np.argmax(y_predict)) #트레이닝 후의 추론값 또한 결과값과 다를 수도 있다.(정확도가 87% 정도이기에)
      return SN

   else:
      print("잘못 입력하셨습니다. 학습을 하지 않습니다.")
      return False


x_train, y_train, x_test, y_test = get_data()

# W값과 Bias 값을 np.random.random 함수를 사용해 알맞게 넣어주십시오.(이 것만 빈칸 나머지는 제공)
# 3.1
W = np.random.random((784,10)) # score function의 결과값 Y는 X*W + b이고, W에는 이미지의 픽셀 크기(784)만큼의 parameter가 class 개수(10)만큼 존재한다.따라서 [784,10]가 W의 크기이고, random하게 값을 입력한다.
Bias = np.random.random((1,10)) #score function의 결과값 Y에서 Bias는 각 class별로 내가 더 선호하거나 싫어하는 정도를 나타내기 위해 특정 값을 더하거나 빼는 형식이다. 따라서 class 개수만큼의 parameter를 가진다. 따라서(1,10)의 크기만큼 존재한다.
#i = input() # 자신이 원하는 숫자 넣기 가능
i = 5
print("train 데이터의 {} 번째의 값 추출".format(i))

Trainend = TestSN(i, x_train, y_train, x_test, y_test, W, Bias) #위의 TestNN함수를 호출해 작업을 돌림.

#밑에 것은 심심하면 자신이 트레이닝한 것이 잘되는지 실험해보세요.
'''
if Trainend !=False:
   TrainNN =Trainend
   print("몇 번 추론하실 겁니까?")
   iterator = input()
   if(is_number(iterator)):
      for i in range(0, int(iterator)):
         print("x_train의 s번째 데이터를 뽑아주세요.\n")
         s = int(input())
         print("S : {}".format(s))
         check = x_train[s]
         img_show(check)
         Hypothesis = TrainNN.Forward(check)
         print("이 이미지의 추론 값 : {}".format(np.argmax(Hypothesis)))
   else:
      print("iterator로 숫자를 안넣었습니다. 종료합니다.")

'''


