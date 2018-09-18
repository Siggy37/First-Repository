# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 11:31:53 2018

@author: brand
"""

import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
#from tensorflow.examples.tutorials.mnist import input_data
#from  import input_data #the place it's from
#input_= input_data.read_data_sets("_________") #wherever it is
#thedata = input_data.read_data_sets()

data_path = 'train.tfrecords'
val_path = 'val.tfrecords'
test_path = 'test.tfrecords'


batch_size=100

feature = {'train/image': tf.FixedLenFeature([], tf.string),
               'train/label': tf.FixedLenFeature([], tf.int64)}
filename_queue = tf.train.string_input_producer([data_path, val_path, test_path], num_epochs=3)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example, features=feature)
image = tf.decode_raw(features['train/image'], tf.float32)
label = tf.cast(features['train/label'], tf.int32)
image = tf.reshape(image, [228384])
images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=30, num_threads=1, min_after_dequeue=10)

n_nodes_hl1=200
n_nodes_hl2=200
n_nodes_hl3=200

size = 228384   #size of an unwrapped picture

classes=10


def neural_network_model(data):
    hidden_layer1={'weights':tf.Variable(tf.random_normal([size,n_nodes_hl1])),
                   'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_layer2={'weights':tf.Variable(tf.random_normal ([n_nodes_hl1, n_nodes_hl2])),
                   'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_layer3={'weights':tf.Variable(tf.random_normal ([n_nodes_hl2, n_nodes_hl3])),
                   'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3, classes])),
                   'biases':tf.Variable(tf.random_normal([classes]))}
    
    l1=tf.add(tf.matmul(data, hidden_layer1['weights']), hidden_layer1['biases'])
    l1=tf.nn.relu(l1)
    
    l2=tf.add(tf.matmul(l1, hidden_layer2['weights']), hidden_layer2['biases'])
    l2=tf.nn.relu(l2)
    
    l3=tf.add(tf.matmul(l2, hidden_layer3['weights']), hidden_layer3['biases'])
    l3=tf.nn.relu(l3)
    
    output=(tf.matmul(l3, output_layer['weights']) + output_layer['biases'])
    
    return output
    
    
    
assert images.shape[0] == labels.shape[0]
x=tf.placeholder(images.dtype, images.shape)
y=tf.placeholder(labels.dtype, images.shape)

dataset = Dataset(100,228384) #code is breaking here, method tensorslices is not working properly @google this is your fault
#dataset = dataset.batch(32)
#dataset = dataset.repeat()
print(dataset)
iterator = dataset.make_initializable_iterator()


def next_batch(num, data, labels):
   # Return a total of `num` random samples and labels. 

    idx = np.arange(0 , 2000)
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 4
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(2000/batch_size)):
                epoch_x, epoch_y = next_batch(batch_size, dataset, labels)
                _, c = sess.run(iterator.initializer, feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch ', epoch, ' completed out of ', hm_epochs, ' loss: ', epoch_loss)
        
        
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy ', accuracy.eval({x:images.test.images, y:images.test.labels}))
        
        
train_neural_network(x)