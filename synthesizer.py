from PIL import Image
import math, wave, array, sys, getopt, glob, os
from scipy.io.wavfile import write

def getData(volume, freq, sampleRate, index):
    return int(volume * math.sin(freq * math.pi * 2 * index /sampleRate))

# pathdir="SoundFile/stft_dir"
# subpath=["/Guitar", "/Piano"]
# subsubpath=["/train","/test"]
# for k in range(0,1):
#     for j in range(0,1):
#         sub=subpath[k] + subsubpath [j]
# path=pathdir + sub
path = "data"
print("path: " + path)
files = [name for name in glob.glob(path + "**/*.png", recursive = True)]
# print(files)
for i_file in range(0,len(files)):

    inputfile=files[i_file]
    print("input: " + inputfile)
    outputfile=inputfile.split('.')[0]
    outputfile=outputfile.split('/')[-1]
    # outputpath="results" + sub + "/" + outputfile
    outputpath = outputfile
    print(outputfile)

    im = Image.open(inputfile)

    width, height = im.size
    rgb_im = im.convert('RGB')

    duration=3.5
    durationSeconds = float(duration) 
    tmpData = []
    maxFreq = 0
    data = array.array('h')
    sampleRate = 22050
    channels = 1
    dataSize = 2 

    numSamples = int(sampleRate * durationSeconds)
    samplesPerPixel = math.floor(numSamples / width)

    C = 20000 / height

    for x in range(numSamples):
        rez = 0
        
        pixel_x = int(x / samplesPerPixel)
        if pixel_x >= width:
            pixel_x = width -1
            
        for y in range(height):
            r, g, b = rgb_im.getpixel((pixel_x, y))
            s = r + g + b
            
            volume = s * 100 / 765
            
            if volume == 0:
                continue
            
            freq = int(C * (height - y + 1))
            
            rez += getData(volume, freq, sampleRate, x)

        tmpData.append(rez)
        if abs(rez) > maxFreq:
            maxFreq = abs(rez)

    for i in range(len(tmpData)):
        data.append(int(32767 * tmpData[i] / maxFreq))
    f = wave.open(outputpath, 'w')
    f.setparams((channels, dataSize, sampleRate, numSamples, "NONE", "Uncompressed"))
    f.writeframes(data.tostring())
    f.close()


