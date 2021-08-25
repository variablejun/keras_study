from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 맛있는 밥을 진짜 마구 마구 먹엇다'
#{'진짜': 1, '나는': 2, '매우': 3, '맛있는': 4, '밥을': 5, '마구마구': 6, '먹엇다': 7}
# 우선순위 1 횟수 2 순차적으로 앞에있는거부터, 7가지 단어를 1부터 7까지 수치화
# 글자들간의 높낮이가 생기기 땐문에 그것을 없애주기 위해 원핫인코딩 실행
token = Tokenizer() # 문자를 수치화시킨다.
token.fit_on_texts([text])
print(token.word_index)

x = token.texts_to_sequences([text])
print(x)
#[[2, 1, 3, 4, 5, 1, 6, 7]]

from tensorflow.keras.utils import to_categorical

word_Size = len(token.word_index)
print(word_Size)

# 7

x  = to_categorical(x)

print(x)
print(x.shape)

'''
[[[0. 0. 0. 1. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 1.]]]
(1, 9, 8)
'''