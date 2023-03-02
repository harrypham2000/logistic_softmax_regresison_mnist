"""
This file is for multiclass fashion-mnist classification using TensorFlow

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from util import get_mnist_data
from logistic_np import add_one
from softmax_np import *
import time
from sklearn.metrics import precision_recall_fscore_support
import tensorflow.compat.v1 as tf2
tf2.disable_v2_behavior()



if __name__ == "__main__":
    np.random.seed(2020)
    tf.random.set_seed(2020)

    # Load data from file
    # Make sure that fashion-mnist/*.gz files is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
    num_train = train_x.shape[0]
    num_val = val_x.shape[0]
    num_test = test_x.shape[0]  

    # generate_unit_testcase(train_x.copy(), train_y.copy()) 

    # Convert label lists to one-hot (one-of-k) encoding
    train_y = create_one_hot(train_y)
    val_y = create_one_hot(val_y)
    test_y = create_one_hot(test_y)

    # Normalize our data
    train_x, val_x, test_x = normalize(train_x, val_x, test_x)
    
    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x) 
    val_x = add_one(val_x)
    test_x = add_one(test_x)
   
    # [TODO 2.8] Create TF placeholders to feed train_x and train_y when training
    x = tf2.placeholder(tf.float32, [None, train_x.shape[1]], name='x')
    y = tf2.placeholder(tf.float32, [None, train_y.shape[1]], name='y')

    # [TODO 2.8] Create weights (W) using TF variables 
    W = tf.Variable(tf.zeros([train_x.shape[1], train_y.shape[1]]), dtype=tf.float32, name='W')

    # [TODO 2.9] Create a feed-forward operator 
    pred = tf.nn.softmax(tf.matmul(x, W))

    # [TODO 2.10] Write the cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

    # Define hyperparameters and train-related parameters
    #num_epoch = 10000
    num_epoch = 100000
    learning_rate = 0.15

    # [TODO 2.8] Create an SGD optimizer
    optimizer = tf2.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # Some meta parameters
    epochs_to_draw = 1000
    all_train_loss = []
    all_val_loss = []
    plt.ion()
    num_val_increase = 0
    prev_val_loss=np.inf
    patience = 5
    # Start training
    init = tf2.global_variables_initializer()

    with tf2.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            tic = time.time()
            # [TODO 2.8] Compute losses and update weights here
            train_loss = sess.run(cost, feed_dict={x:train_x, y:train_y})
            val_loss = sess.run(cost, feed_dict={x:val_x, y:val_y})
            _, train_loss = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})
            val_loss = sess.run(cost, feed_dict={x: val_x, y: val_y})
            # Update weights
            sess.run(optimizer, feed_dict={x: train_x, y: train_y})
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            toc = time.time()
            if (e % epochs_to_draw == epochs_to_draw-1):
                plot_loss(all_train_loss,all_val_loss)
                plt.savefig(f'/content/drive/MyDrive/HW2_ML/HW2_ML/plot_softmax_tf_{e}.png')
                plt.show()
                plt.pause(0.1)
                plt.close()
            print(f"Epoch:{e},time={toc - tic:4f},train_loss={train_loss:4f},val_loss={val_loss:4f}")
            # [TODO 2.11] Define your own stopping condition here
            if val_loss >= prev_val_loss:
                num_val_increase+=1
            else:
                num_val_increase=0
            prev_val_loss=val_loss
            if num_val_increase>=patience:
                print('Stopping training. Validation loss has not improved for {} epochs.'.format(num_val_increase))
                break
        
        y_hat = sess.run(pred, feed_dict={x: test_x})
        test(y_hat, test_y)
