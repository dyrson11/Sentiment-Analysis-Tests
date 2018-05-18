from sklearn.utils import shuffle
import Word_Embedding_1 as WE
import datetime
import tensorflow as tf
import numpy as np

# CARGA DE LOS CANALES
FastText, Word2Vec, GloVe = WE.loadChannels()
data = WE.loadData_2_in("./Task_1/intertass-train-tagged.xml", "./Task_1/intertass-development-tagged.xml") # oraciones para el entrenmiento
data_test = WE.loadDataTesting("./Task_1/intertass-test.xml", "./Task_1/intertass-sentiment.qrel") # oraciones para el testing

#data = WE.loadData_1_in("./Task_1/general-train-tagged-3l.xml") # oraciones para el entrenmiento
#data_test = WE.loadDataTesting("./Task_1/general-test-tagged-3l.xml", "./Task_1/general-sentiment-3l.qrel") # oraciones para el testing

x_3d, y_d, max_mat_train = WE.extract_data(data, FastText, Word2Vec, GloVe)
x_test, y_test, max_mat_test = WE.extract_data(data_test, FastText, Word2Vec, GloVe)
max_mat = max(max_mat_train, max_mat_test)

# DATA DE TRAINING
x_3d, y_d = shuffle(x_3d, y_d, random_state=0)
x_3d = WE.decode_we(x_3d, max_mat, FastText, Word2Vec, GloVe) #input pre-procesada
dic_size = x_3d.shape[2]

# DATA DE TESTING
x_test = WE.decode_we(x_test, max_mat, FastText, Word2Vec, GloVe) #input pre-procesada

# CREACION DE LA RED
sess = tf.InteractiveSession() #Incialización de la session
# x_input = tf.placeholder(tf.float32, shape=[None, max_mat, dic_size, 3]) # 3 channels
x_input = tf.placeholder(tf.float32, shape=[None, max_mat, dic_size]) # 1 channel
y_output = tf.placeholder(tf.float32, shape=[None, y_d.shape[1]])

def weight_values(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_values(shape):
  return tf.Variable(tf.constant(0.1, shape=shape))

def conv_2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, dic_size, 1], padding='VALID')

def max_pool_2x1(x, b_):
  return tf.nn.max_pool(x, ksize=[1, b_, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

def feature_maps(x, c, dic, m, n):
    for i in range(len(c)):
        #W_conv = weight_values([c[i], dic, 3, n]) for 3 channels
        W_conv = weight_values([c[i], dic, 1, n])
        b_conv = bias_values([n])
        h_conv = tf.nn.relu(conv_2d(x, W_conv) + b_conv)
        h_pool = max_pool_2x1(h_conv, m+1-c[i])
        h_pool_flat = tf.reshape( h_pool,[-1,n])
        if i == 0:
            fc_input = h_pool_flat
        else:
            fc_input = tf.concat([fc_input, h_pool_flat], 1)
    return fc_input

dropout_keep_prob = tf.placeholder(tf.float32)
hidden_units = 100

# LSTM Layer
cellFwd = tf.contrib.rnn.LSTMCell(hidden_units, state_is_tuple=True)
cellFwd = tf.contrib.rnn.DropoutWrapper(cell=cellFwd, output_keep_prob=dropout_keep_prob)
cellBwd = tf.contrib.rnn.LSTMCell(hidden_units, state_is_tuple=True)
cellBwd = tf.contrib.rnn.DropoutWrapper(cell=cellBwd, output_keep_prob=dropout_keep_prob)

outputs, final_rnn_state = tf.nn.bidirectional_dynamic_rnn(cellFwd, cellBwd, x_input, dtype=tf.float32)
merged = tf.concat(outputs, 2)
output = tf.transpose(merged, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)

v_att = tf.tanh(tf.matmul(tf.reshape(last, [-1, hidden_units*2]), self.w_att) \
                    + tf.reshape(self.b_att, [1, -1]))
        betas = tf.matmul(v_att, tf.reshape(self.u_att, [-1, 1]))

        exp_betas = tf.reshape(tf.exp(betas), [-1, self.maxSeqLength])
        alphas = exp_betas / tf.reshape(tf.reduce_sum(exp_betas, 1), [-1, 1])

        output = tf.reduce_sum(hidden_layer * tf.reshape(alphas,
                                                         [-1, self.maxSeqLength, 1]), 1)



## Output
W_fc2 = weight_values([hidden_units*2, y_d.shape[1]])
b_fc2 = bias_values([y_d.shape[1]])
y = tf.nn.softmax(tf.matmul(last, W_fc2) + b_fc2)

## Training - Testing
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_output, logits=y))
train_function = tf.train.AdamOptimizer(1e-4).minimize(cost) # Update rule
#train_function = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#n_fragments = 11 #cross_validation
batch_size = 55
n_epoch = 2300

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for i in range(n_epoch):
        x_train, y_train = WE.get_minibatch(batch_size, i, x_3d, y_d)
        sess.run(train_function, feed_dict={x_input: x_train[:,:,:,0], y_output: y_train, dropout_keep_prob: 0.5})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_input: x_train[:,:,:,0], y_output: y_train, dropout_keep_prob: 0.5})
            print('step %d, eficiencia del training %g' % (i, train_accuracy))

    # testing
    ef_test = accuracy.eval(feed_dict={x_input: x_test[:,:,:,0], y_output: y_test, dropout_keep_prob: 1})
    print('eficiencia del test %g' % ef_test)

    name = "modelos_task_1/" + str(ef_test) + "/pretrained.ckpt"
    save_path = saver.save(sess, name)
    #saver.restore(sess, "modelos/variable.ckpt")

'''
# CROSS VALIDATION
with tf.Session() as sess:
    for it in range(n_fragments):
        sess.run(tf.global_variables_initializer())
        x_3d_train, y_d_train, x_3d_test, y_d_test= WE.cross_validation(it, n_fragments, x_3d, y_d)
        for i in range(n_epoch):
            x_train, y_train = WE.get_minibatch(batch_size, i, x_3d_train, y_d_train)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x_input: x_train, y_output: y_train, keep_prob: 0.5})
                print('Paso %d, eficiencia del training %g' % (i, train_accuracy))
            train_function.run(feed_dict={x_input: x_train, y_output: y_train, keep_prob: 0.5})

        # testing
        ef = accuracy.eval(feed_dict={x_input: x_3d_test, y_output: y_d_test, keep_prob: 1})
        print('######## Fragmento %d, eficiencia = %g' % (it, ef))'''
