import tensorflow as tf
import numpy as np 

class Parameter:
    ### general structure ######
    
    def __init__(self,depth=[]):
        self.depth = depth
        
        
    def __call__(self):

        with tf.variable_scope("param"):
            with tf.variable_scope("fc1"):
                weight1 = tf.get_variable()
                bias1 = tf.get_variable()

            with tf.variable_scope("fc2"):
                weight2 = tf.get_variable()
                bias2 = tf.get_variable()

            with tf.variable_scope("fc3"):
                weight = tf.get_variable()
                bias = tf.get_variable()

            with tf.variable_scope("residual_1"):
                pass
            with tf.variable_scope("residual_2"):
                pass
            with tf.variable_scope("residual_3"):
                pass
            with tf.variable_scope("fc4"):
                pass
            with tf.variable_scope("fc5"):
                pass
            
                
                

        

    
        
