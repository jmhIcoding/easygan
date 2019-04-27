__author__ = 'jmh081701'
import  tensorflow as tf
import  data_gen
import  numpy as np
from matplotlib import  pyplot as plt
dator = data_gen.data_generator()

batch=200
k = 5

realpoint_x = tf.placeholder(dtype=tf.float32,shape=[None,2],name="realpoint")
noise_z =  tf.placeholder(dtype=tf.float32,shape=[None,1],name="noise")

#生成器
g_layers=[1,32,64,128,2]
#第一层
Wg_1 = tf.Variable(tf.truncated_normal(shape=[g_layers[0],g_layers[1]],stddev=0.1),name="Wg1")
bg_1 = tf.Variable(tf.zeros(shape=[g_layers[1]]),"biasg1")
#第二层
Wg_2 = tf.Variable(tf.truncated_normal(shape=[g_layers[1],g_layers[2]],stddev=0.1),name="Wg2")
bg_2 = tf.Variable(tf.zeros(shape=[g_layers[2]]),"biasg2")
#第三层
Wg_3 = tf.Variable(tf.truncated_normal(shape=[g_layers[2],g_layers[3]],stddev=0.1),name="Wg3")
bg_3 = tf.Variable(tf.zeros(shape=[g_layers[3]]),"biasg3")
#输出层
Wg_4 = tf.Variable(tf.truncated_normal(shape=[g_layers[3],g_layers[4]],stddev=0.1),name="Woutputg")
bg_4 = tf.Variable(tf.zeros(shape=[g_layers[4]]),"Woutputg")

def G(noise):
    print(type(noise))
    z = noise
    featureMapg1 = tf.nn.relu(tf.matmul(z,Wg_1)+ bg_1 )
    featureMapg2 = tf.nn.sigmoid(tf.matmul(featureMapg1,Wg_2)+bg_2)
    featureMapg3 = tf.nn.relu(tf.matmul(featureMapg2,Wg_3)+bg_3)
    fake_x = 6*(tf.nn.sigmoid(tf.matmul(featureMapg3,Wg_4)+bg_4) -0.5)
    return fake_x

#鉴别器
d_layers=[2,16,64,80,1]
#第一层的权值矩阵
Wd_1 = tf.Variable(tf.truncated_normal(shape=[d_layers[0],d_layers[1]],stddev=0.1),name="Wd1")
bd_1 = tf.Variable(tf.zeros(shape=[d_layers[1]]),"biasd1")

#第二层的权值矩阵
Wd_2 = tf.Variable(tf.truncated_normal(shape=[d_layers[1],d_layers[2]],stddev=0.1),name="Wd2")
bd_2 = tf.Variable(tf.zeros(shape=[d_layers[2]]),"biasd2")

#第三层的权值矩阵
Wd_3 = tf.Variable(tf.truncated_normal(shape=[d_layers[2],d_layers[3]],stddev=0.1),name="Wd3")
bd_3 = tf.Variable(tf.zeros(shape=[d_layers[3]]),"biasd3")

#输出层的权值矩阵
Wd_4 = tf.Variable(tf.truncated_normal(shape=[d_layers[3],d_layers[4]],stddev=0.1),name="Woutputd")
bd_4 = tf.Variable(tf.zeros(shape=[d_layers[4]]),"Woutputd")

def D(input_x):
    x = input_x
    #x = tf.reshape(input_x,shape=[None,2])
    featureMapd1 = tf.nn.relu(tf.matmul(x,Wd_1)+ bd_1 )
    featureMapd2 = tf.nn.tanh(tf.matmul(featureMapd1,Wd_2)+bd_2)
    featureMapd3 = tf.nn.relu(tf.matmul(featureMapd2,Wd_3)+bd_3)
    score= tf.nn.sigmoid(tf.matmul(featureMapd3,Wd_4)+bd_4)
    return score


fake_x = G(noise_z)

d_loss = -tf.reduce_mean(tf.log(D(realpoint_x)) + tf.log(1-D(fake_x)))
g_loss = tf.reduce_mean(tf.log(1-D(fake_x)))

#使用var_list来指定只更新部分参数
d_learner = tf.train.AdamOptimizer(learning_rate=0.001).minimize(d_loss,var_list=[Wd_1,Wd_2,Wd_3,Wd_4,bd_1,bd_2,bd_3,bd_4])

g_learner = tf.train.AdamOptimizer(learning_rate=0.001).minimize(g_loss,var_list=[Wg_1,Wg_2,Wg_3,Wg_4,bg_1,bg_2,bg_3,bg_4])


if __name__ == '__main__':
    max_epoch = 1000
    step = 1
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        while step< 30000:
            for _k in range(k):
                input_points = dator.next_batch(batch)
                noises = np.random.normal(size=[batch,1])
                loss,learner= sess.run(fetches=[d_loss,d_learner],feed_dict={realpoint_x:input_points,noise_z:noises})
            input_points=dator.next_batch(batch)
            noises = np.random.normal(size=[batch,1])
            loss,learner = sess.run(fetches=[g_loss,g_learner],feed_dict={realpoint_x:input_points,noise_z:noises})
            if step % 100 ==0:
                print({'step':step,'loss':loss,'epoch':dator.epoch})
            step +=1
        print("##############TEST#################")
        noises = np.random.normal(size=[batch,1])
        fake_point = sess.run(fetches=[fake_x],feed_dict = {noise_z:noises})[0]
        print(fake_point)
        plt.scatter(x=fake_point[:,0],y=fake_point[:,1],c='g',marker='+')

        noises = np.random.normal(size=[batch,1])
        fake_point = sess.run(fetches=[fake_x],feed_dict = {noise_z:noises})[0]
        print(fake_point)
        plt.scatter(x=fake_point[:,0],y=fake_point[:,1],c='g',marker='*')
        plt.show()


