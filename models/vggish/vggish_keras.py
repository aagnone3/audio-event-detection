from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model, Sequential

from models.vggish import vggish_params
from models.base import BaseModel


class Vggish(BaseModel):

    defaults = {
        "learning_rate": 0.001,
        "definition": "functional"
    }

    def __init__(self, input_dim, n_classes, **kwargs):
        super(Vggish, self).__init__(**kwargs)
        self.input_dim = input_dim
        print("Input dim: {}".format(self.input_dim))
        self.n_classes = n_classes
        self.build_seq_model()

    def vggish_keras_functional(self):
        input_shape = (vggish_params.NUM_FRAMES,vggish_params.NUM_BANDS, 1)
        print("Vggish-defined input shape: {}".format(input_shape))
        img_input = Input(shape=self.input_dim)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1', trainable=False)(img_input)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1', trainable=False)(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2', trainable=False)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2', trainable=False)(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1', trainable=False)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2', trainable=False)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3', trainable=False)(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1', trainable=False)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2', trainable=False)(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4', trainable=False)(x)

        # Block fc
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1_1', trainable=False)(x)
        x = Dense(4096, activation='relu', name='fc1_2', trainable=False)(x)
        x = Dense(vggish_params.EMBEDDING_SIZE, activation='relu', name='fc2', trainable=False)(x)

        # classification layers
        x = Dense(256, activation='relu', name='clf_1')(x)
        x = Dropout()(x)
        x = Dense(self.n_classes, activation='relu', name='clf_1')(x)

        model = Model(img_input, x, name='vggish')
        return model


    def vggish_keras_sequential(self):
        input_shape = (vggish_params.NUM_FRAMES,vggish_params.NUM_BANDS,1)

        # Block 1
        model = Sequential([
            Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1', input_shape=input_shape),
            MaxPooling2D((2, 2), strides=(2, 2), name='pool1'),

            # Block 2
            Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2'),
            MaxPooling2D((2, 2), strides=(2, 2), name='pool2'),

            # Block 3
            Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'),
            Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'),
            MaxPooling2D((2, 2), strides=(2, 2), name='pool3'),

            # Block 4
            Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'),
            Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'),
            MaxPooling2D((2, 2), strides=(2, 2), name='pool4'),

            # Block fc
            Flatten(name='flatten'),
            Dense(4096, activation='relu', name='fc1_1'),
            Dense(4096, activation='relu', name='fc1_2'),
            Dense(vggish_params.EMBEDDING_SIZE, activation='relu', name='fc2')
        ], name='vggish')

        return model
