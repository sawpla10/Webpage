import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.models import Model


# 사용자의 수와 스터디 그룹의 수
num_users = 1000
num_study_groups = 500

# 사용자의 선호도와 스터디 그룹의 특성을 입력으로 받는 레이어
user_input = Input(shape=(1,))
study_group_input = Input(shape=(1,))

# 임베딩 레이어
user_embedding = Embedding(num_users, 50)(user_input)
study_group_embedding = Embedding(num_study_groups, 50)(study_group_input)

# 임베딩 레이어의 출력을 평탄화
user_embedding = Flatten()(user_embedding)
study_group_embedding = Flatten()(study_group_embedding)

# 사용자의 임베딩과 스터디 그룹의 임베딩을 내적하여 예측 점수를 계산
score = Dot(axes=1)([user_embedding, study_group_embedding])

# 모델을 생성하고 컴파일
model = Model(inputs=[user_input, study_group_input], outputs=score)
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습 (여기서는 임의의 데이터를 사용하였습니다.)
user_ids = tf.random.uniform((10000, 1), minval=0, maxval=num_users, dtype=tf.int32)
study_group_ids = tf.random.uniform((10000, 1), minval=0, maxval=num_study_groups, dtype=tf.int32)
ratings = tf.random.uniform((10000, 1))
model.fit([user_ids, study_group_ids], ratings, epochs=5)
