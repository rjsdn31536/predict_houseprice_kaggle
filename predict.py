import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# 처리해놓은 데이터 로딩
train_file = "./data/train_2.csv"
test_file = "./data/test_2.csv"

x_train = pd.read_csv(train_file, engine='python')
x_test = pd.read_csv(test_file, engine='python')

y_train = x_train.SalePrice
x_train.drop(['SalePrice'], axis=1, inplace=True)

y_train_log = np.log(y_train)

x_train_np = x_train.as_matrix()
# y_train_np = y_train_log.as_matrix()
x_test_np = x_test.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(x_train_np,
                                                    y_train_log,
                                                    test_size=0.33,
                                                    random_state=7)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.distplot(y_train, bins=50)
plt.title('Original Data')
plt.xlabel('Sale Price')

plt.subplot(1,2,2)
sns.distplot(y_train_log, bins=50)
plt.title('Natural Log of Data')
plt.xlabel('Natural Log of Sale Price')
plt.tight_layout()


import tensorflow as tf

num_unit1 = 200
# num_unit2 = 75
keepout = 0.5
step = 11000
learning_rate = 0.003

'''
    x1 : Input
    W1 : weight
    b1 : bias
'''
# Input layer
x1 = tf.placeholder(tf.float32, [None, X_train.shape[1]])
W1 = tf.Variable(tf.truncated_normal([X_train.shape[1], num_unit1],
                                          stddev=0.1))
b1 = tf.Variable(tf.constant(1., shape = [num_unit1]))

# Hidden1 layer
hidden1 = tf.nn.relu(tf.matmul(x1, W1) + b1)

keep_prob = tf.placeholder(tf.float32)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)


# Input layer
# W2 = tf.Variable(tf.truncated_normal([num_unit1, num_unit2],
#                                           stddev=0.1))
# b2 = tf.Variable(tf.constant(1., shape = [num_unit2]))
#
# # Hidden2 layer
# hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, W2) + b2)
#
# hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

W0 = tf.Variable(tf.truncated_normal([num_unit1, 1],
                                          stddev=0.1))
b0 = tf.Variable(tf.constant(1., shape = [1]))

# Output layer(Labels)
k = tf.matmul(hidden1_drop, W0) + b0
y_train.shape

y_ = tf.placeholder(tf.float32, [None,1])
cost = tf.losses.mean_squared_error(labels = y_, predictions = k)
# cost = tf.reduce_mean(tf.square(k - y_))
# cost = tf.reduce_sum((y_ - k))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

print ("Training\n")
sess = tf.Session()
init = tf.global_variables_initializer() #.run()
sess.run(init)

# tensorboard를 사용하기 위하여 summary.scalar를 사용
tf.summary.scalar('loss', cost)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./tensorboard', sess.graph)

# (?, ) -> (?, 1)
y_train = y_train.reshape([y_train.shape[0], 1])

j=0

for _ in range(step):
    j += 1
    summary, _ = sess.run([merged, train_step],
                        feed_dict={x1: X_train, y_: y_train, keep_prob:0.5})
    train_writer.add_summary(summary, j)
    if j % 100 == 0 :
        print('step :', j, end='\t')
        print ('loss :',sess.run(cost, feed_dict={x1: X_train, y_: y_train, keep_prob:keepout}))

train_writer.close()



print ("Testing model")
# Test trained model
prediction = np.exp(sess.run(k, feed_dict={x1: X_test, keep_prob : 1}))

print('hidden layer : {}, step : {}'.format(num_unit1, step))
print('rmsle')
print(rmsle(prediction, np.exp(y_test)))

submission = pd.DataFrame()
submission['Id'] = x_test.index+1461
submission['SalePrice'] = (prediction)
print(submission)
submission.to_csv('./output_csv/MLP_tensorflow_all_normalize_skewed.csv', index=False)