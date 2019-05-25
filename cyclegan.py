from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.optimizers import RMSprop
from keras.models import Model
from keras.models import load_model
from keras.layers.merge import concatenate
from keras.utils import multi_gpu_model
from keras.models import load_model


from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.utils import plot_model

import numpy as np
import argparse
import datetime
from glob import glob
import os
import cv2

import librosa
import librosa.display
import matplotlib.pyplot as plt

def encoder_layer(inputs,
                  filters=16,
                  kernel_size=(9,1),
                  strides=(2,1),
                  activation='relu',
                  instance_norm=True):
    """Builds a generic encoder layer made of Conv2D-IN-LeakyReLU
    IN is optional, LeakyReLU may be replaced by ReLU

    """

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)

    return x


def decoder_layer(inputs,
                  paired_inputs,
                  filters=16,
                  kernel_size=(9,1),
                  strides=(2,1),
                  activation='relu',
                  instance_norm=True):
    """Builds a generic decoder layer made of Conv2D-IN-LeakyReLU
    IN is optional, LeakyReLU may be replaced by ReLU
    Arguments: (partial)
    inputs (tensor): the decoder layer input
    paired_inputs (tensor): the encoder layer output
          provided by U-Net skip connection &
          concatenated to inputs.

    """

    conv = Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    x = concatenate([x, paired_inputs])

    return x

def build_generator(input_shape,
                    output_shape=None,
                    kernel_size=(9,1),
                    name=None):
    """The generator is a U-Network made of a 4-layer encoder
    and a 4-layer decoder. Layer n-i is connected to layer i.

    Arguments:
    input_shape (tuple): input shape
    output_shape (tuple): output shape
    kernel_size (int): kernel size of encoder & decoder layers
    name (string): name assigned to generator model

    Returns:
    generator (Model):

    """

    inputs = Input(shape=input_shape)
    channels = int(output_shape[-1])
    e1 = encoder_layer(inputs,
                       32,
                       kernel_size=kernel_size,
                       activation='leaky_relu',
                       strides=1)
    e2 = encoder_layer(e1,
                       64,
                       activation='leaky_relu',
                       kernel_size=kernel_size)
    e3 = encoder_layer(e2,
                       128,
                       activation='leaky_relu',
                       kernel_size=kernel_size)
    e4 = encoder_layer(e3,
                       256,
                       activation='leaky_relu',
                       kernel_size=kernel_size)

    d1 = decoder_layer(e4,
                       e3,
                       128,
                       kernel_size=kernel_size)
    d2 = decoder_layer(d1,
                       e2,
                       64,
                       kernel_size=kernel_size)
    d3 = decoder_layer(d2,
                       e1,
                       32,
                       kernel_size=kernel_size)
    outputs = Conv2DTranspose(channels,
                              kernel_size=kernel_size,
                              strides=1,
                              activation='sigmoid',
                              padding='same')(d3)

    generator = Model(inputs, outputs, name=name)
    generator=keras.utils.multi_gpu_model(generator, gpus=2)

    return generator

def build_discriminator(input_shape,
                        kernel_size=(9,1),
                        patchgan=True,
                        name=None):
    """The discriminator is a 4-layer encoder that outputs either
    a 1-dim or a n x n-dim patch of probability that input is real

    Arguments:
    input_shape (tuple): input shape
    kernel_size (int): kernel size of decoder layers
    patchgan (bool): whether the output is a patch or just a 1-dim
    name (string): name assigned to discriminator model

    Returns:
    discriminator (Model):

    """

    inputs = Input(shape=input_shape)
    x = encoder_layer(inputs,
                      32,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    x = encoder_layer(x,
                      64,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    x = encoder_layer(x,
                      128,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    x = encoder_layer(x,
                      256,
                      kernel_size=kernel_size,
                      strides=(1,1),
                      activation='leaky_relu',
                      instance_norm=False)

    # if patchgan=True use nxn-dim output of probability
    # else use 1-dim output of probability
    if patchgan:
        x = LeakyReLU(alpha=0.2)(x)
        outputs = Conv2D(1,
                         kernel_size=kernel_size,
                         strides=(2,1),
                         padding='same')(x)
    else:
        x = Flatten()(x)
        x = Dense(1)(x)
        outputs = Activation('linear')(x)


    discriminator = Model(inputs, outputs, name=name)
    discriminator = keras.utils.multi_gpu_model(discriminator, gpus=2)

    return discriminator

def train_cyclegan(models, data, params, test_params, test_generator):
    """ Trains the CycleGAN.

    1) Train the target discriminator
    2) Train the source discriminator
    3) Train the forward and backward cyles of adversarial networks

    Arguments:
    models (Models): Source/Target Discriminator/Generator,
                     Adversarial Model
    data (tuple): source and target training data
    params (tuple): network parameters
    test_params (tuple): test parameters
    test_generator (function): used for generating predicted target
                    and source images
    """
    # the models
    g_source, g_target, d_source, d_target, adv = models
    # network parameters
    batch_size, train_steps, patch, model_name = params
    # train dataset
    source_data, target_data, test_source_data, test_target_data = data

    titles, dirs = test_params
    dir_pred_source, dir_pred_target = dirs
    # the generator image is saved every 2000 steps
    
    save_interval = 2000
    length = min(len(test_source_data), len(test_target_data))
    test_indexes = np.random.randint(0, length, size=16)
    target_size = target_data.shape[0]
    source_size = source_data.shape[0]

    samples = test_source_data[test_indexes]
    stftdisplay(samples,
            filename='test_source_data',
            samples_dir=dir_pred_source,
            title='test_source',
            show=False)
    samples = test_target_data[test_indexes]
    stftdisplay(samples,
            filename='test_target_data',
            samples_dir=dir_pred_target,
            title='test_target',
            show=False)
    # whether to use patchgan or not
    if patch > 1:
        d_patch = (patch, patch, 1)
        valid = np.ones((batch_size,) + d_patch)
        fake = np.zeros((batch_size,) + d_patch)
    else:
        valid = np.ones([batch_size, 1])
        fake = np.zeros([batch_size, 1])

    valid_fake = np.concatenate((valid, fake))
    start_time = datetime.datetime.now()

    for step in range(train_steps):
        # sample a batch of real target data
        rand_indexes = np.random.randint(0, target_size, size=batch_size)
        real_target = target_data[rand_indexes]

        # sample a batch of real source data
        rand_indexes = np.random.randint(0, source_size, size=batch_size)
        real_source = source_data[rand_indexes]
        # generate a batch of fake target data fr real source data
        fake_target = g_target.predict(real_source)

        # combine real and fake into one batch
        x = np.concatenate((real_target, fake_target))
        # train the target discriminator using fake/real data
        metrics = d_target.train_on_batch(x, valid_fake)
        log = "%d/%d: [d_target loss: %f]" % (step+1, train_steps, metrics[0])

        # generate a batch of fake source data fr real target data
        fake_source = g_source.predict(real_target)
        x = np.concatenate((real_source, fake_source))
        # train the source discriminator using fake/real data
        metrics = d_source.train_on_batch(x, valid_fake)
        log = "%s [d_source loss: %f]" % (log, metrics[0])

        # train the adversarial network using forward and backward
        # cycles. the generated fake source and target data attempts
        # to trick the discriminators
        x = [real_source, real_target]
        y = [valid, valid, real_source, real_target]
        metrics = adv.train_on_batch(x, y)
        elapsed_time = datetime.datetime.now() - start_time
        fmt = "%s [adv loss: %f] [time: %s]"
        log = fmt % (log, metrics[0], elapsed_time)
        print(log)
        if (step + 1) % save_interval == 0:
            if (step + 1) == train_steps:
                show = True
            else:
                show = False

            test_generator((g_source, g_target),
                           (test_source_data, test_target_data),
                           step=step+1,
                           titles=titles,
                           dirs=dirs,
                           test_indexes=test_indexes,
                           show=show)

            # save the models after training the generators
            g_source.save("models/"+str(step+1) + "steps_" + model_name + "-g_source.h5")
            g_target.save("models/"+str(step+1) + "steps_" + model_name + "-g_target.h5")

def test_generator(generators, test_data, step,
                   titles, dirs, test_indexes=range(16),
                   show=False):
    """Test the generator models

    Arguments:
    generators (tuple): source and target generators
    test_data (tuple): source and target test data
    step (int): step number during training (0 during testing)
    titles (tuple): titles on the displayed image
    dirs (tuple): folders to save the outputs of testings
    todisplay (int): number of images to display (must be
        perfect square)
    show (bool): whether to display the image or not
          (False during training, True during testing)

    """


    # predict the output from test data
    g_source, g_target = generators
    test_source_data, test_target_data = test_data
    t1, t2, t3, t4 = titles
    title_pred_source = t1
    title_pred_target = t2
    title_reco_source = t3
    title_reco_target = t4
    dir_pred_source, dir_pred_target = dirs

    pred_target_data = g_target.predict(test_source_data)
    pred_source_data = g_source.predict(test_target_data)
    reco_source_data = g_source.predict(pred_target_data)
    reco_target_data = g_target.predict(pred_source_data)


    # display the 1st todisplay images
    samples = pred_target_data[test_indexes]
    # print(len(samples))
    # print(np.shape(samples))
    # print(samples[0])
    filename = '%06d.png' % step
    step = "Step{:,}".format(step)
    title = title_pred_target + step
    stftdisplay(samples,
            filename=filename,
            samples_dir=dir_pred_target,
            title=title,
            show=show)

    samples = pred_source_data[test_indexes]
    title = title_pred_source + step
    stftdisplay(samples,
            filename=filename,
            samples_dir=dir_pred_source,
            title=title,
            show=show)

    samples = reco_source_data[test_indexes]
    title = title_reco_source
    filename = "reconstructed_source.png"
    stftdisplay(samples,
            filename=filename,
            samples_dir=dir_pred_source,
            title=title,
            show=show)

    samples = reco_target_data[test_indexes]
    title = title_reco_target
    filename = "reconstructed_target.png"
    stftdisplay(samples,
            filename=filename,
            samples_dir=dir_pred_target,
            title=title,
            show=show)

def wav_spec(samples,
            filename,
            title='',
            samples_dir=None,
            show=False):

    sr = 2000
    sample_length = 8000

    # create saved_images folder
    if samples_dir is None:
        samples_dir = 'saved_images'
    save_dir = os.path.join(os.getcwd(), samples_dir, title)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(samples_dir, filename)

    fig , axes = plt.subplots(4,4)
    fig.suptitle(title)
    fig.set_size_inches(18, 9)
    for sample, ax, i in zip(samples, axes.flat, range(16)):
        sample.resize(sample_length)
        audiofile = '%s/%s-%d.wav' % (save_dir,title, i)
        librosa.output.write_wav(audiofile, sample, sr)
        spec = librosa.stft(sample)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
        librosa.display.specshow(db, sr=sr, ax=ax, x_axis='time', y_axis='linear')

    plt.savefig(filename)
    if show:
        plt.show()

    plt.close('all')

def stftdisplay(samples, filename, title='',
            samples_dir=None, show=False):
    if samples_dir is None:
        samples_dir = 'saved_images'
    save_dir = os.path.join(os.getcwd(), samples_dir, title)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # Saving each spectogram
    for i in range(len(samples)):
        filename = os.path.join(save_dir,'%02d.png'%i)
        cv2.imwrite(filename,samples[i]*255)
    
    # filename = os.path.join(samples_dir, filename)

    # fig , axes = plt.subplots(4,4)
    # fig.suptitle(title)
    # fig.set_size_inches(18, 9)

    # for sample, ax, i in zip(samples, axes.flat, range(16)):
    #     ax.xaxis.set_visible(False)
    #     ax.yaxis.set_visible(False)
    #     ax.imshow(sample)

    # plt.savefig(filename)
    # if show:
    #     plt.show()

    # plt.close('all')

def build_cyclegan(shapes,
                   source_name='source',
                   target_name='target',
                   kernel_size=3,
                   patchgan=False,
                   identity=False
                   ):
    """Build the CycleGAN

    1) Build target and source discriminators
    2) Build target and source generators
    3) Build the adversarial network

    Arguments:
    shapes (tuple): source and target shapes
    source_name (string): string to be appended on dis/gen models
    target_name (string): string to be appended on dis/gen models
    kernel_size (int): kernel size for the encoder/decoder or dis/gen
                       models
    patchgan (bool): whether to use patchgan on discriminator
    identity (bool): whether to use identity loss

    Returns:
    (list): 2 generator, 2 discriminator, and 1 adversarial models

    """

    source_shape, target_shape = shapes
    lr = 2e-4
    decay = 6e-8
    gt_name = "gen_" + target_name
    gs_name = "gen_" + source_name
    dt_name = "dis_" + target_name
    ds_name = "dis_" + source_name

    # build target and source generators
    g_target = build_generator(source_shape,
                               target_shape,
                               kernel_size=kernel_size,
                               name=gt_name)
    g_source = build_generator(target_shape,
                               source_shape,
                               kernel_size=kernel_size,
                               name=gs_name)
    print('---- TARGET GENERATOR ----')
    g_target.summary()
    print('---- SOURCE GENERATOR ----')
    g_source.summary()

    # build target and source discriminators
    d_target = build_discriminator(target_shape,
                                   patchgan=patchgan,
                                   kernel_size=kernel_size,
                                   name=dt_name)
    d_source = build_discriminator(source_shape,
                                   patchgan=patchgan,
                                   kernel_size=kernel_size,
                                   name=ds_name)
    print('---- TARGET DISCRIMINATOR ----')
    d_target.summary()
    print('---- SOURCE DISCRIMINATOR ----')
    d_source.summary()

    optimizer = RMSprop(lr=lr, decay=decay)
    d_target.compile(loss='mse',
                     optimizer=optimizer,
                     metrics=['accuracy'])
    d_source.compile(loss='mse',
                     optimizer=optimizer,
                     metrics=['accuracy'])

    d_target.trainable = False
    d_source.trainable = False

    # build the computational graph for the adversarial model
    # forward cycle network and target discriminator
    source_input = Input(shape=source_shape)
    fake_target = g_target(source_input)
    preal_target = d_target(fake_target)
    reco_source = g_source(fake_target)

    # backward cycle network and source discriminator
    target_input = Input(shape=target_shape)
    fake_source = g_source(target_input)
    preal_source = d_source(fake_source)
    reco_target = g_target(fake_source)

    # if we use identity loss, add 2 extra loss terms
    # and outputs
    if identity:
        iden_source = g_source(source_input)
        iden_target = g_target(target_input)
        loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae']
        loss_weights = [1., 1., 10., 10., 0.5, 0.5]
        inputs = [source_input, target_input]
        outputs = [preal_source,
                   preal_target,
                   reco_source,
                   reco_target,
                   iden_source,
                   iden_target]
    else:
        loss = ['mse', 'mse', 'mae', 'mae']
        loss_weights = [1., 1., 10., 10.]
        inputs = [source_input, target_input]
        outputs = [preal_source,
                   preal_target,
                   reco_source,
                   reco_target]

    # build adversarial model
    adv = Model(inputs, outputs, name='adversarial')
    adv=keras.utils.multi_gpu_model(adv, gpus=2)
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    adv.compile(loss=loss,
                loss_weights=loss_weights,
                optimizer=optimizer,
                metrics=['accuracy'])
    print('---- ADVERSARIAL NETWORK ----')
    adv.summary()

    # os.makedirs('models/base-cgan', exist_ok=True)
    # plot_model(g_source, to_file='models/base-cgan/g_source.png', show_shapes=True)
    # plot_model(g_target, to_file='models/base-cgan/g_target.png', show_shapes=True)
    # plot_model(d_source, to_file='models/base-cgan/d_source.png', show_shapes=True)
    # plot_model(d_target, to_file='models/base-cgan/d_target.png', show_shapes=True)
    # plot_model(adv, to_file='models/base-cgan/adv.png', show_shapes=True)
    return g_source, g_target, d_source, d_target, adv

def load_data(source_name='Piano',
            target_name='Guitar',
            sample_length=8000
            ):
    # data = {}
    # domain_cats = []
    # for domain in ['valid', 'test']:
    #     path = './data/nsynth-processed-%s/' % domain
    #     for npy_file in sorted(glob(os.path.join(path, '*.npy'))):
    #         cat = npy_file[len(path) + len(domain) + 1: -4]
    #         if cat in [source_name, target_name]:
    #             domain_cat = '%s-%s' % (domain,cat)
    #             domain_cats.append(domain_cat)
    #             data[domain_cat] = []
    #             data[domain_cat] = np.load(npy_file)
    #             print(npy_file[len(path) : ], np.shape(data[domain_cat]))
    # for domain in ['test','train']:
    #     path = './stft_dir/%s' %domain
    #     for imgs in sorted(glob(os.path.join(path, '*.png'))):
    #         cat = imgs[len(path) + len(domain) + 1]
    #         if cat in [source_name, target_name]:
    source_data, test_source_data, target_data, test_target_data = [], [], [], []
    for imgs in sorted(glob('./stft_data/Piano/train/*.png')):
        imgA = cv2.imread(imgs,1).astype('float32')/255
        source_data.append(np.asarray(imgA))
    print(np.shape(source_data))

    for imgs in sorted(glob('./stft_data/Piano/test/*.png')):
        imgA = cv2.imread(imgs,1).astype('float32')/255
        test_source_data.append(np.asarray(imgA))
    print(np.shape(test_source_data))

    for imgs in sorted(glob('./stft_data/Guitar/train/*.png')):
        imgA = cv2.imread(imgs,1).astype('float32')/255
        target_data.append(np.asarray(imgA))
    print(np.shape(target_data))

    for imgs in sorted(glob('./stft_data/Guitar/test/*.png')):
        imgA = cv2.imread(imgs,1).astype('float32')/255
        test_target_data.append(np.asarray(imgA))
    print(np.shape(test_target_data))
             
    # source_data = np.reshape(data[domain_cats[0]], (len(data[domain_cats[0]]),sample_length,1,1))
    # target_data = np.reshape(data[domain_cats[1]], (len(data[domain_cats[1]]),sample_length,1,1))
    # test_source_data = np.reshape(data[domain_cats[2]], (len(data[domain_cats[2]]),sample_length,1,1))
    # test_target_data = np.reshape(data[domain_cats[3]], (len(data[domain_cats[3]]),sample_length,1,1))

    data = (source_data, target_data, test_source_data, test_target_data)
    return data

def nsynth(g_models=None):
    """
    # keyboard_electronic <--> guitar_acoustic
    piano_acoustic <-> guitar_acoustic
    """
    cats = [
            'bass_electronic', 'bass_synthetic', 'brass_acoustic', 'flute_acoustic',
            'flute_synthetic', 'guitar_acoustic', 'guitar_electronic',
            'keyboard_acoustic', 'keyboard_electronic', 'keyboard_synthetic',
            'mallet_acoustic', 'organ_electronic', 'reed_acoustic', 'string_acoustic',
            'vocal_acoustic', 'vocal_synthetic'
            ]

    sample_length = 8000

    source_shape = (480, 640, 3)
    target_shape = (480, 640, 3)

   
    shapes = (source_shape, target_shape)
    kernel_size = (9,1)
    source_name = 'piano_acoustic'
    target_name = 'guitar_acoustic'

    source_data = np.load('./stft_dir/%s-train.npy' % source_name)
    target_data = np.load('./stft_dir/%s-train.npy' % target_name)
    test_source_data = np.load('./stft_dir/%s-test.npy' % source_name)
    test_target_data = np.load('./stft_dir/%s-test.npy' % target_name)

    print(np.shape(source_data), np.shape(target_data), np.shape(test_source_data), np.shape(test_target_data))
    data = (source_data, target_data, test_source_data, test_target_data)
    model_name = 'cyclegan_ka_ga'
    batch_size = 1
    train_steps = 20000
    patchgan = False

    # patch size is divided by 2^n since we downscaled the input
    # in the discriminator by 2^n (ie. we use strides=2 n times)
    patch = int(source_data.shape[1] / 2**4) if patchgan else 1

    postfix = ('9x1p') if patchgan else ('9x1')
    titles = ('%s_predicted-source' % source_name,
             '%s_predicted-target' % target_name,
             '%s_reconstructed-source' % source_name,
             '%s_reconstructed-target' % target_name)
    dirs = ('%s_source-%s' % (source_name,postfix), '%s_target-%s' % (target_name,postfix))

    params = (batch_size, train_steps, patch, model_name)
    test_params = (titles, dirs)

    models = build_cyclegan(shapes,
                       source_name=source_name,
                       target_name=target_name,
                       kernel_size=kernel_size,
                       patchgan=patchgan
                       )

    train_cyclegan(models,
                data,
                params,
                test_params,
                test_generator)

if __name__ == '__main__':
    nsynth()
