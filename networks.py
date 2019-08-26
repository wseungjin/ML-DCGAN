from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np

class DCGAN(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.phase = args.phase
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset

        self.epoch = args.epoch
        self.iteration = args.iteration
        
        self.lr = args.lr

        
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq        
        
        self.img_size = args.img_size
        self.img_ch = args.img_ch
        
        self.ch = args.ch
        
        
        
        self.c_dim = 3
        self.data = load_data(dataset_name=self.dataset_name)
        self.custom_dataset = True

        self.dataset_num = len(self.data)
        
        print()

        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        print("# dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)

        print()
        print("# learning rate : ", self.lr)
            
    def gernertaor(self, x_init, reuse=False, scope="gernerator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x= conv(x_init,channel*2,kernel=2,stride=1,pad=3,scope="conv1")
            x= batch_norm(x)
            x= relu(x)
            
            x= conv(x,channel*2,kernel=2,stride=2,pad=3,scope="conv2")
            x= batch_norm(x)
            x= relu(x)
            
            x= conv(x,channel//2,kernel=2,stride=2,pad=3,scope="conv3")
            x= batch_norm(x)
            x= relu(x)

            x= conv(x,channel//2,kernel=2,stride=2,pad=3,scope="conv4")
            x= batch_norm(x)
            x= relu(x)
            
            return x
            
        
        
    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x= conv(x_init,channel*2,kernel=2,stride=1,pad=3,scope="conv1")
            x= batch_norm(x)
            x= relu(x)
            
            x= conv(x,channel*2,kernel=2,stride=2,pad=3,scope="conv2")
            x= batch_norm(x)
            x= relu(x)
            
            x= conv(x,channel//2,kernel=2,stride=2,pad=3,scope="conv3")
            x= batch_norm(x)
            x= relu(x)

            x= conv(x,channel//2,kernel=2,stride=2,pad=3,scope="conv4")
            x= batch_norm(x)
            x= relu(x)
            
            x= flatten(x)
            x= fully_connected(x,channel)
            
            return x
        
        
    def loss():
        
            
        
    def build_model(self):
        if self.phase == 'train' :
            self.lr = tf.placeholder(tf.float32, name='learning_rate')
            
            image_Data_Class = ImageData(self.img_size, self.img_ch)
            
            train = tf.data.Dataset.from_tensor_slices(self.train_dataset)
            
            gpu_device = '/gpu:0'
            train = train.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))

            train_iterator = train.make_one_shot_iterator()
            
            self.domain = train_iterator.get_next()
            
            
    

        
    def train():
    
    def model_dir(self):
        if self.sn :
            sn = '_sn'
        else :
            sn = ''

        return "{}_{}_{}_{}_{}_{}{}".format(
            self.model_name, self.dataset_name, self.gan_type, self.img_size, self.z_dim, self.moment, sn)
    
    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
        
    def test(): 
    
    