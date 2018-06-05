#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 10:47:31 2018

@author: yonatank
"""
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras.backend as K


#input_tensor = graph.get_tensor_by_name('dense_1_input:0')
#graph.get_operations()
#tf.global_variables() 
# [ op for op in graph.get_operations() if op.type == "Placeholder"]
#[ op for op in graph.get_operations() if "loss" in op.name]

def make_model():
    model = Sequential()
    model.add(Dense(32, input_dim=10,activation='relu'))#input 10 features
    model.add(Dense(1, activation='sigmoid'))#input dim
    
    model.compile(optimizer='sgd',loss='mse')    
    return model

def test_loss(model):
    train_x = np.random.rand(1024,10)
    train_y = np.random.rand(1024,1)
    sample_weights = np.random.rand(1024)
    
    sess = K.get_session()
    graph = sess.graph
    P = []
    [P.append(op) for op in graph.get_operations() if op.type == "Placeholder"]
    input_tensor = graph.get_tensor_by_name("dense_1_input:0")
    weights_tensor = graph.get_tensor_by_name("dense_2_sample_weights:0")
    target_tensor = graph.get_tensor_by_name("dense_2_target:0")
    loss_tensor = graph.get_tensor_by_name("loss/mul:0")
    loss_np = sess.run(loss_tensor,feed_dict={input_tensor:train_x,
        weights_tensor:sample_weights,target_tensor:train_y})
    
    print("loss shape:{},val:{}".format(loss_np.shape,loss_np))
    
    
    
    


def test_mode():    
    train_x = np.random.rand(1024,10)
    train_y = np.random.rand(1024,1)
    sample_weights = np.random.rand(1024,1)
    model = make_model()
    model.fit(train_x,train_y)
    train_pred = model.predict(train_x)
    print("Train pred:{}".format(train_pred))
    
    
if __name__ == "__main__":
    test_mode()

    

    
    
