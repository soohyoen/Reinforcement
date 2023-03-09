발산하는 이유 :

gradient loss → sigmoid함수 (gradient 가 0이 되기때문에 vanishing gradient)
              → sigmoid를 썼을때 아주작은 값을 제곱하면 더 작은값이 되어버려 저속 수렴문제가 발생
              → mse는 단순히 거리를 비교하며, cross entropy는 두개의 확률분포간의 거리를 비교

→ activation function : softmax, loss function : cross entropy
→  learning_rate 가 너무 낮으면 발산가능성
→  dropout : depth가 깊어지면 overfiting가능성 이를 사용하여 과적합을 막음

이전 상황 :
Dropout을 사용하니 애초에 수렴이 안되는 문제가 발생

시도 2:
● activation function : softmax, loss function : cross entropy 사용

기존 relu에서 softmax로 바꿈.

loss function mse → cross entropy

mse : 평균 제곱 오차 손실
        → 신경망의 출력과 타겟이 연속인 회귀 문제에서 널리 사용하는 손실함수
          예 측과 타겟값의 차이를 제곱하여 평균한 값
 
categorical_crossentropy : 범주형 교차 엔트로피
                           → 출력을 클래스 소속 확률에 대한 예측으로 이해할 수 있는 문제에서 사용
                              (레이블(y) 클래스가 2개 이상일 경우 사용, 멀티클래스 분류에 사용)
                           → 활성화 함수는 softmax(벡터의 요소값에서 0~1사이값에서 모든 합이 1이 됨)

이론상으로 cross entropy는 맞아 보이지않음. 실제 학습 결과로 전혀 수렴하지 않음.

last layer가 softmax였네.. 이전 Dense는 relu와 마지막 Dense는 softmax로 바꿔서 학습

softmax : 출력값이 N개
          입력값을 각각 지수함수로 취하고, 이를 정규화
          정규화로 인해 각 출력값은 0~1 값을 가짐
          모든 출력값의 합은 반드시 1
          N가지 중 한가지에 속할 확률 표현 가능(=Multi-class Classification)
          softmax를 사용하는 경우 결과를 확률로 변경하며 결과가 분류를 해야되는 것이라면 softmax를 
          써서 모든 결과가 1로 수렴하게 만들 수 있음.

결과적으로 cross entropy와 softmax를 쓰니까 수렴함. 학습결과도 좋음.. 

Env를 바꿔서 시험한 결과 전혀 수렴하지않음.. 이게 무슨일이람
