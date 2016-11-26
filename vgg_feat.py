import numpy as np
import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from imagenet_utils import preprocess_input

weights_path_th = 'vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
weights_path_tf = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG16(include_top=False, weights='imagenet',
          input_tensor=None):
    
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    model = Model(img_input, x1)
    if K.image_dim_ordering() == 'th':
        model.load_weights(weights_path_th,by_name=True)
    else:
        model.load_weights(weights_path_tf,by_name=True)

    return model

if __name__ == "__main__":
    model = VGG16()
    img_path = 'health.png'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    pred = model.predict(x)
    # from keras import backend as K

    # # with a Sequential model
    # get_3rd_layer_output = K.function([model.layers[0].input],
    #                                   [model.layers[3].output])
    # preds = get_3rd_layer_output([x])[0]