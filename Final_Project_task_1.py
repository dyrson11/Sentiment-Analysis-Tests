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
x_input = tf.placeholder(tf.float32, shape=[None, max_mat, dic_size, 3]) # 3 channels
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
        W_conv = weight_values([c[i], dic, 3, n])
        b_conv = bias_values([n])
        h_conv = tf.nn.relu(conv_2d(x, W_conv) + b_conv)
        h_pool = max_pool_2x1(h_conv, m+1-c[i])
        h_pool_flat = tf.reshape(h_pool,[-1,n])
        if i == 0:
            fc_input = h_pool_flat
        else:
            fc_input = tf.concat([fc_input, h_pool_flat], 1)
    return fc_input

# Convolution Network - Pooling - paralelas
conv_def = [1 ,2, 3, 4]
out_conv_ch = 100
hidden_units = 200

fc_input = feature_maps(x_input, conv_def, dic_size, max_mat, out_conv_ch)
W_fc1 = weight_values([len(conv_def)*out_conv_ch, hidden_units])
b_fc1 = bias_values([hidden_units])
h_fc1 = tf.nn.relu(tf.matmul(fc_input, W_fc1) + b_fc1)

## DropOut
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## Output
W_fc2 = weight_values([hidden_units, y_d.shape[1]])
b_fc2 = bias_values([y_d.shape[1]])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

## Training - Testing
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_output, logits=y))
train_function = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # Update rule
#train_function = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#n_fragments = 11 #cross_validation
batch_size = 55
n_epoch = 2400

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for i in range(n_epoch):
        x_train, y_train = WE.get_minibatch(batch_size, i, x_3d, y_d)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_input: x_train, y_output: y_train, keep_prob: 0.5})
            print('step %d, eficiencia del training %g' % (i, train_accuracy))
        train_function.run(feed_dict={x_input: x_train, y_output: y_train, keep_prob: 0.5})

    # testing
    ef_test = accuracy.eval(feed_dict={x_input: x_test, y_output: y_test, keep_prob: 1})
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
