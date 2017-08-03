import tensorflow as tf
import numpy as np 

class Parameter:
    ### general structure ######
    
    def __init__(self,depth=[],input_size):
        self.depth = depth
        self.input_size = input_size
        
    def __call__(self,input_vector):

        with tf.variable_scope("param"):
            with tf.variable_scope("fc1"):
                weight1 = tf.get_variable("w1",[input_size,self.depth[0]],tf.random_uniform_initialiazer())
                bias1 = tf.get_variable("b1",[self.depth[0]])
                fc1 = tf.add(tf.matmul(input_vector,weight1),bias1)
                fc1 = tf.nn.relu(fc1)

            with tf.variable_scope("fc2"):
                weight2 = tf.get_variable("w2",[self.depth[0],self.depth[1]],tf.random_uniform_initializer())
                bias2 = tf.get_variable("b2",[self.depth[1]],tf.random_uniform_initializer())
                fc2 = tf.add(tf.matmul(fc1,weight2),bias2)
                fc2 = tf.nn.relu(fc2)
            with tf.variable_scope("fc3"):
                weight3 = tf.get_variable("w3",[self.depth[1],self.depth[2]],tf.random_uniform_initializer())
                bias3 = tf.get_variable("b2",[self.depth[2]],tf.random_uniform_initializer())
                fc3 = tf.add(tf.matmul(fc2,weight3),bias3)
                fc3 = tf.nn.relu(fc3)

            with tf.variable_scope("residual_1"):
                res1_w1 = tf.get_variable("res1_w",[self.depth[2],self.depth[2]],tf.random_uniform_initializer())
                res1_b1 = tf.get_variable("res1_b",[self.depth[2]],tf.random.uniform_initializer())
                res1 = tf.add(tf.matmul(fc3,res_w1),res1_b1)
                res1 = res1 + fc3
                res1 = tf.nn.relu(res1)
            with tf.variable_scope("residual_2"):
                res2_weight = tf.get_variable("res2_w",[self.depth[2],self.depth[2]],tf.random.uniform_initializer())
                res2_bias = tf.get_variable("res2_b",[self.depth[2]],tf.random_uniform_initializer())
                res2 = tf.add(tf.matmul(res1,res2_weight),res2_bias)
                res2 = res2 + res1
            
                res2 = tf.nn.relu(res2)
            with tf.variable_scope("residual_3"):
                res3_weight = tf.get_variable("res3_w",[self.depth[2],self.depth[2]],tf.random_uniform_initializer())
                res3_bias = tf.get_variable("res3_b",[self.depth[2]],tf.random.uniform_initializer())
                res3 = tf.add(tf.matmul(res2,res3_weight),res3_bias)
                res3 = res3 + res2
                
                res3 = tf.nn.relu(res3)
            with tf.variable_scope("fc4"):
                weight4 = tf.get_variable("w4",[self.depth[2],self.depth[3]],tf.random.uniform_initializer())
                bias4 = tf.get_variable("b4",[self.depth[3]],tf.random.uniform_initializer())
                fc4 = tf.add(tf.matmul(res3,weight4),bias4)
                fc4 = tf.nn.relu(fc4)
            with tf.variable_scope("fc5"):
                weight5 = tf.get_variable("w5",[self.depth[3],self.depth[4]],tf.random.uniform_initializer())
                bias5 = tf.get_variable("b5",[self.depth[4]],tf.random.uniform_initializer())

                fc5 = tf.add(tf.matmul(fc4,weight5),bias5)
                
         
                

        

    
        
