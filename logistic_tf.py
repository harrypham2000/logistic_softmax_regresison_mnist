"""
This file is for binary classification using TensorFlow
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tf

from util import get_vehicle_data
from logistic_np import *
import tensorflow.compat.v1 as tf2
tf2.disable_v2_behavior()

if __name__ == "__main__":
    np.random.seed(2018)
    tf.random.set_seed(2018)

    # Load data from file
    # Make sure that vehicles.dat is in data/
    train_x, train_y, test_x, test_y = get_vehicle_data()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]

    #generate_unit_testcase(train_x.copy(), train_y.copy())
    # logistic_unit_test()

    # Normalize our data: choose one of the two methods before training
    #train_x, test_x = normalize_all_pixel(train_x, test_x)
    train_x, test_x = normalize_per_pixel(train_x, test_x)

    # Reshape our data
    # train_x: shape=(2400, 64, 64) -> shape=(2400, 64*64)
    # test_x: shape=(600, 64, 64) -> shape=(600, 64*64)
    train_x = reshape2D(train_x)
    test_x = reshape2D(test_x)

    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x)
    test_x = add_one(test_x)

    # [TODO 1.11] Create TF placeholders to feed train_x and train_y when training
    x=tf2.placeholder(tf.float32,[None,train_x.shape[1]],name='x')
    y=tf2.placeholder(tf.float32,[None,1],name='y')
    # [TODO 1.12] Create weights (W) using TF variables
    W=tf.Variable(tf.zeros((train_x.shape[1],1)),dtype=tf.float32,name='W')

    # [TODO 1.13] Create a feed-forward operator
    pred=tf.sigmoid(tf.matmul(x,W))
    #Why use tf.reduce_sum? Because working with 2D input instead of 3D input, and we want to compute only the weights
    #with each x input so the * will calculate the whole matrix and TensorFlow cannot define whether it's a 1D or 2D matrix
    #reduce_sum will calculate the sum of each weights with the vector.
    # [TODO 1.14] Write the cost function
    #cost = -tf.reduce_sum(y*tf.log(pred)+(1-y)*tf.log(1-pred))/num_train
    cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred))

    # Define hyperparameters and train-related parameters
    num_epoch = 10000
    learning_rate = 0.1
#    momentum_rate = 0.9

    # [TODO 1.15] Create an SGD optimizer
    optimizer=tf2.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # Some meta parameters
    epochs_to_draw = 1000
    all_loss = []
    plt.ion()

    # Start training
    init = tf2.global_variables_initializer()

    with tf2.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            tic = time.time()
            # [TODO 1.16] Compute loss and update weights here

            # Update weights...
            _, loss = sess.run([optimizer, cost], feed_dict={x:train_x, y:train_y})
            all_loss.append(loss)

            if (e % epochs_to_draw == epochs_to_draw-1):
                plot_loss(all_loss)
                plt.savefig(f'/content/drive/MyDrive/HW2_ML/HW2_ML/plot_logistic_tf_{e}.png')
                plt.show()
                plt.pause(0.1)
                plt.close()
            print("Epoch %d: loss is %.5f" % (e+1, loss))
            toc = time.time()
            print("Time:",toc-tic)
        y_hat = sess.run(pred, feed_dict={x: test_x})
        test(y_hat, test_y)

