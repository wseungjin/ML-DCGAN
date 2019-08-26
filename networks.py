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
        
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr

        
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq        
        
        self.img_size = args.img_size
        self.img_ch = args.img_ch
        
        self.ch = args.ch
        
        self.gan_type = args.gan_type
        self.z_dim = args.z_dim
        
        self.c_dim = 3
        self.data = load_data(dataset_name=self.dataset_name)
        self.custom_dataset = True

        self.dataset_num = len(self.data)
        
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        
        print()

        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        print("# dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# iteration per epoch : ", self.iteration)

        print("##### Generator #####")
        print("# learning rate : ", self.g_learning_rate)

        print()

        print("##### Discriminator #####")
        print("# learning rate : ", self.d_learning_rate)
            
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
        loss
            
        
    def build_model(self):
        """ Graph Input """
        # images
        Image_Data_Class = ImageData(self.img_size, self.c_dim, self.custom_dataset)
        inputs = tf.data.Dataset.from_tensor_slices(self.data)
        
        gpu_device = '/gpu:0'
        inputs = inputs.\
            apply(shuffle_and_repeat(self.dataset_num)).\
            apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).\
            apply(prefetch_to_device(gpu_device, self.batch_size))
        
        inputs_iterator = inputs.make_one_shot_iterator()

        self.inputs = inputs_iterator.get_next()
        
        # noises
        self.z = tf.random_normal(shape=[self.batch_size, 1, 1, self.z_dim], name='random_z')
        
        """ Loss Function """
        # output of D for real images
        real_logits = self.discriminator(self.inputs)

        # output of D for fake images
        fake_images = self.generator(self.z)
        fake_logits = self.discriminator(fake_images, reuse=True)
        
        # get loss for discriminator
        self.d_loss = discriminator_loss(self.gan_type, real=real_logits, fake=fake_logits)

        # get loss for generator
        self.g_loss = generator_loss(self.gan_type, fake=fake_logits)

        t_vars = tf.trainable_variables()
        for var in t_vars: 
            if 'discriminator' in var.name:
                d_vars = [var]
        for var in t_vars: 
            if 'generator' in var.name:
                g_vars = [var]
            
    

        
    def train():
        train
        
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
        test
    
    
    def generate_image():
        generate