import tensorflow as tf
import numpy as np
import parameter
 
class STYLE:
    def __init__(self,conv_depth=[32,64,128],deconv_depth=[64,32,3],epsilon = 1e-3):
        self.conv_depth = conv_depth
        self.deconv = deconv_depth
        self.epsilon = epsilon
    def __call__(self,input_img):
        with tf.variables_scope("style"):
            with tf.variable_scope("conv_1"):
                conv1 = tf.layers.conv2d(input_img,self.conv_depth[0],kernel_size=[9,9],strides=(1,1),padding="SAME")
                ##### shift and scale is from parameter network #########
                #self.instance_norm()
                conv1 = tf.nn.relu(conv1)
            with tf.variable_scope("conv2"):
                conv2 = tf.layers.conv2d(conv1,self.conv_depth[1],kernel_size =[3,3],strides=(2,2),padding="SAME")
                ###### Shift and scale is from parameter network
                #self.instance_norm()
                conv2 = tf.nn.relu(conv2)
            with tf.variable_scope("conv3"):
                conv3 = tf.layers.conv2d(conv2,self.conv_depth[2],kernel_size=[3,3],strides = (2,2),padding="SAME")
                ######shift and scale is from parameter network #####
                #conv2 = self.instance_norm()
                conv3 = tf.nn.relu(conv3)
            with tf.variable_scope("residual_1"):
                res1 = tf.layers.conv2d(conv3,self.conv_depth[2],kernel_size=[3,3],strides=(1,1),padding="SAME")
                res1 = conv3 + res1
            with tf.variable_scope("residual_2"):
                res2 = tf.layers.conv2d(res1,self.conv_depth[2],kernel_size=[3,3],stride=(1,1),padding="SAME")
                res2 = res1+ res2
            with tf.variable_scope(residual_3):
                res3 = tf.layers.conv2d(res2,self.conv_depth[2],kernel_size=[3,3],stride=(1,1),padding="SAME")
                res3 = res2 + res3

            with tf.variable_scope("deconv_1"):
                deconv1 = tf.layers.conv2d_transpose(res3,self.deconv_depth[0],kernel_size=[3,3],padding="SAME")
                ###### instance normalization shift scale is from parameter ####
                # deconv1 = self.instance_norm()
                deconv1 = tf.nn.relu(deconv1)
            with tf.variable_scope("deconv_2"):
                deconv2 = tf.layers.conv2d_transpose(deconv1,self.deconv_depth[1],kernel_size=[3,3],padding="SAME")
                ##### read parameter.py  scale shift #####
                #self.instance_norm()
                deconv2 = tf.nn.relu(deconv2)
            with tf.variable_scope("deconv3"):
                deconv3 = tf.layers.conv2d_transpose(deconv2,self.deconv_depth[2],kernel_size=[3,3],padding="SAME")
                ######## read parameter.py scale shift #####
                #deconv3 = self.instance_norm(deconv3)
            return deconv3                
    def instance_norm(self,shift,scale,input_tensor):
        mu,sigma = tf.nn.moments(input_tenser,[1,2],keep_dims = True) 
        normalized = (input_tensor - mu)/(sigma + self.epsilon)**(0.5) 
        
        return scale * normalized + shift
