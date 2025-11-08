import keras_hub
import numpy as np
import tensorflow as tf
import scipy.stats as stats
import tifffile as tif
import os
import sys

batch_size = 1

def make_tensor_batch(mip, indices):
    img = np.zeros((len(indices), 512, 512, 3), np.float32)
    for i, index in enumerate(indices):
        rfp_zscore = stats.zscore(mip[index, 1], axis=None)
        img[i] = np.stack([rfp_zscore] * 3, axis=-1)
    return tf.convert_to_tensor(img)

def make_label_batch(model, tensor_batch):
    labels = model.predict(tensor_batch)
    return np.argmax(labels, axis=3)

def main():

    #load model
    model = keras_hub.models.ImageSegmenter.from_preset('deeplab_v3_plus_resnet50_pascalvoc', num_classes=2, compile=False)
    model.load_weights('deeplab_051525.weights.h5')

    #load mip
    mip_path = sys.argv[1]
    name = os.path.splitext(os.path.basename(mip_path))[0]
    mip = tif.imread(mip_path)

    #predict in batches
    output_dir = os.path.dirname(mip_path)
    output_path = os.path.join(output_dir, 'label.tif')
    output = np.zeros((len(mip), 512, 512), bool)
    for i in range(0, len(mip), batch_size):
        batch_range = range(i, min(i + batch_size, len(mip)))
        tensor = make_tensor_batch(mip, batch_range)
        label = make_label_batch(model, tensor)
        output[batch_range] = label
        print(f'{name}: {i}')
    tif.imwrite(output_path, output)
