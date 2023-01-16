발산하는 이유 :

gradient loss → sigmoid함수 (gradient 가 0이 되기때문에 vanishing gradient)
              → sigmoid를 썼을때 아주작은 값을 제곱하면 더 작은값이 되어버려 저속 수렴문제가 발생
              → mse는 단순히 거리를 비교하며, cross entropy는 두개의 확률분포간의 거리를 비교

→ activation function : softmax, loss function : cross entropy
→  learning_rate 가 너무 낮으면 발산가능성
→  dropout : depth가 깊어지면 overfiting가능성 이를 사용하여 과적합을 막음


시도 1 : 
● activation function : softmax, loss function : cross entropy 사용
● dropout 사용
   → stack overflow : 지도학습이 아닌 연속적인 알고리즘의 형태이기 때문에 dropout을 사용하는 것은 옳지 않다.
   → 학습 결과 1 : Dense, Dropout, Dense, Dense, compile로 갔을 때 수렴조차 하지않음
   → 학습 결과 2 : Dense, Dense, Drpoout, Dense, compile 이와 같은 순서도 수렴조차 하지않음

시도 2:
● activation function : softmax, loss function : cross entropy 사용
