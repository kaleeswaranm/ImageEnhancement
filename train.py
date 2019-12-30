from load_data import data_loader
from build_networks import generator_network, discriminator_network, end_to_end_gan, content_loss
from utils import make_dir
from tensorflow.keras.optimizers import Adam
import numpy as np

def train(batch_size, image_size, steps, ckpt_path):

    make_dir(ckpt_path)

    optim = Adam(lr = 0.01)

    generator_net = generator_network(image_size)
    discriminator_net = discriminator_network(image_size)
    
    generator_net.compile(optim, loss=content_loss)
    discriminator_net.compile(optim, loss="binary_crossentropy")
    
    end_to_end = end_to_end_gan(discriminator_net, generator_net, image_size)
    end_to_end.compile(optim, loss=[content_loss, 'binary_crossentropy'], loss_weights=[1.,1e-3])
    
    data_iterator = data_loader(batch_size)
    
    for step in range(steps):

        real_labels = np.ones((batch_size,1))
        fake_labels = np.zeros((batch_size,1))

        correct_batch, degraded_batch = next(data_iterator)

        gen_hr = generator_net.predict(degraded_batch)
        
        discriminator_net.trainable = True

        disc_loss_1 = discriminator_net.train_on_batch(correct_batch, real_labels)
        disc_loss_2 = discriminator_net.train_on_batch(gen_hr, fake_labels)
        disc_loss = 0.5 * np.add(disc_loss_1, disc_loss_2)

        discriminator_net.trainable = False

        end_loss = end_to_end.train_on_batch(degraded_batch, [correct_batch, real_labels])

        print('step: ' + str(step))
        print('discriminator loss: ', disc_loss)
        print('end_loss: ', end_loss)

        if step % 100 == 0:
            generator_net.save(ckpt_path+'/generator_'+str(step)+'.h5')

if __name__ == '__main__':

    batch_size = 4
    image_size = (250,250, 3)
    steps = 10000
    generator_ckpts = 'generator_checkpoints'

    train(batch_size, image_size, steps, generator_ckpts)