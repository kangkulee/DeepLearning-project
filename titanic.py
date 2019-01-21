import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Cython.Shadow import inline
from sympy.physics.quantum.circuitplot import matplotlib

matplotlib
inline

raw_data = pd.read_excel('./titanic.xls')
# raw_data.info()

# 생존자 파이차트 그리기
# f,ax=plt.subplots(1,2,figsize=(12,6))
#
# raw_data['survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.2f%%',ax=ax[0])
# ax[0].set_title('Survived')
# ax[0].set_ylabel('')
#
# sns.countplot('survived',data=raw_data,ax=ax[1])
# ax[1].set_title('Survived')
# plt.show()
# 탑승자 연령 히스토그램
# raw_data['age'].hist(bins=20,figsize=(18,8),grid=False)
# plt.show()
# 서로 연관 있어 보이는 데이터가 무엇인지 상관계수를 찾기
# plt.figure(figsize=(10, 10))
# sns.heatmap(raw_data.corr(), linewidths=0.01, square=True,
#             annot=True, cmap=plt.cm.viridis, linecolor="white")
# plt.title('Correlation between features')
# plt.show()

tmp = []
for each in raw_data['sex']:
    if each == 'female':
        tmp.append(1)
    elif each == 'male':
        tmp.append(0)
    else:
        tmp.append(np.nan)

raw_data['sex'] = tmp

raw_data['survived'] = raw_data['survived'].astype('float')
raw_data['pclass'] = raw_data['pclass'].astype('float')
raw_data['sex'] = raw_data['sex'].astype('float')
raw_data['sibsp'] = raw_data['sibsp'].astype('float')
raw_data['parch'] = raw_data['parch'].astype('float')
raw_data['fare'] = raw_data['fare'].astype('float')

raw_data = raw_data[raw_data['age'].notnull()]
raw_data = raw_data[raw_data['sibsp'].notnull()]
raw_data = raw_data[raw_data['parch'].notnull()]
raw_data = raw_data[raw_data['fare'].notnull()]

raw_data.info()

x_data = raw_data.values[:, [0,3,4,5,6,8]]
y_data = raw_data.values[:, [1]]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.1, random_state=7)
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense

np.random.seed(7)

model = Sequential()
model.add(Dense(255, input_shape=(6,), activation='relu'))
model.add(Dense((1), activation='sigmoid'))
model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])
model.summary()

import IPython.display
from keras.utils.vis_utils import model_to_dot

IPython.display.SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500)

plt.figure(figsize=(12,8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['loss','val_loss', 'acc','val_acc'])
plt.show()

dicaprio = np.array([3., 0., 19., 0., 0., 5.]).reshape(1,6) #남주인공
winslet = np.array([1., 1., 17., 1., 2., 100.]).reshape(1,6) #여주인공

model.predict(dicaprio)
model.predict(winslet)