import tensorflow as tf
import numpy as np 

class Parameter:
    ### general structure ######
    
    def __init__(self,depth=[]):
        self.depth = depth
        
        
    def __call__(self,input_vector):

        with tf.variable_scope("param"):
            with tf.variable_scope("fc1"):
                weight1 = tf.get_variable()
                bias1 = tf.get_variable()
                fc1 = tf.add(tf.matmul(input_vector,weight1),bias1)
                fc1 = tf.nn.relu(fc1)

            with tf.variable_scope("fc2"):
                weight2 = tf.get_variable()
                bias2 = tf.get_variable()
                fc2 = tf.add(tf.matmul(fc1,weight2),bias2)
                fc2 = tf.nn.relu(fc2)
            with tf.variable_scope("fc3"):
                weight3 = tf.get_variable()
                bias3 = tf.get_variable()
                fc3 = tf.add(tf.matmul(fc2,weight3),bias3)
                fc3 = tf.nn.relu(fc3)

            with tf.variable_scope("residual_1"):
                res1 = tf.get_variable("res1",[])
                
            with tf.variable_scope("residual_2"):
                pass
            with tf.variable_scope("residual_3"):
                pass
            with tf.variable_scope("fc4"):
                pass
            with tf.variable_scope("fc5"):
                pass
            
                
                

        

    
        
