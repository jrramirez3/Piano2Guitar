import os
from glob import glob
import numpy as np
from scipy.io import wavfile
#import librosa
#import librosa.display
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2

cats = [
        'bass_electronic', 'bass_synthetic', 'brass_acoustic', 'flute_acoustic',
        'flute_synthetic', 'guitar_acoustic', 'guitar_electronic',
        'keyboard_acoustic', 'keyboard_electronic', 'keyboard_synthetic',
        'mallet_acoustic', 'organ_electronic', 'reed_acoustic', 'string_acoustic',
        'vocal_acoustic', 'vocal_synthetic'
        ]

fs = 16000
sr = 2000
true_length = 64000
sample_length = 8000

def process_nsynth(domain='test'):
    data = {}
    for cat in cats:
        data[cat] = []

    path = './nsynth-%s/audio/' % domain
    print('Loading data from %s ...' % path)

    for audiofile in sorted(glob(os.path.join(path, '*.wav'))):
        filename = audiofile[len(path):]
        cat = filename[ : -16]
        audio_array, _ = librosa.load(audiofile, sr=fs/8)
        print(filename, len(audio_array))
        data[cat].append(audio_array)

    print('Done loading!\n')

    path_processed = 'data/nsynth-processed-%s' % domain
    os.makedirs(path_processed, exist_ok=True)

    for cat in cats:
        npy_file = '%s/%s-%s' % (path_processed, domain, cat)
        print('Saving data[%s] in %s ...' % (cat, npy_file))
        print('shape:', np.shape(data[cat]))
        np.save(npy_file, data[cat])
        print('Saved!')

    print('\n')

def load_processed(domain='test'):
    data = {}
    for cat in cats:
        data[cat] = []

    path = './data/nsynth-processed-%s/' % domain
    for npy_file in sorted(glob(os.path.join(path, '*.npy'))):
        cat = npy_file[len(path) + len(domain) + 1: -4]
        data[cat] = np.load(npy_file)
        print(npy_file[len(path) : ], np.shape(data[cat]))

    return data

def wav_spec(samples,
            filename,
            title='',
            samples_dir=None,
            show=False):


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

data = []
path = './stft_dir/Piano/test/'
# path = './stft_dir/Piano/test/'
for img in sorted(glob(os.path.join(path, '*.png'))):
    stft = cv2.imread(img,1).astype('float32')/255
    data.append(np.asarray(stft))

print(np.shape(data))
np.save('stft_dir/piano_acoustic-test', data)
# np.save('stft_dir/piano_acoustic-test', data)

# load = np.load('stft_dir/guitar_acoustic-test.npy')
# print(np.shape(load))
