import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Input, Conv2D, MaxPooling2D, LSTM, Reshape, Conv2DTranspose, TimeDistributed, Flatten, Dense, UpSampling2D
from tensorflow.keras.models import Model


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ConvLSTM2D, BatchNormalization, Dropout, concatenate, Activation


def segmentation_model(input_shape=(5, 256, 256, 3), num_classes=6):
    # Start with the input
    inputs = Input(shape=input_shape)

    # Encode each image
    encoded_imgs = []
    for i in range(input_shape[0]):
        img_input = inputs[:, i]

        # Encoder
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        encoded_imgs.append(x)

    # Concatenate the features from all images
    x = Concatenate(axis=-1)(encoded_imgs)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)


    x = Conv2D(num_classes, (3, 3), activation=None, padding='same')(x)
    outputs = tf.keras.layers.Lambda(collapse_to_index)(x)

    model = Model(inputs=inputs, outputs=x)
    return model

def build_vgg16_segmentation_bn(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder (VGG16 structure with BatchNormalization)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x1 = MaxPooling2D((2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x2 = MaxPooling2D((2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same')(x2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x3 = MaxPooling2D((2, 2))(x)

    # Decoder with BatchNormalization

    # Block 4
    x = UpSampling2D(size=(2, 2))(x3)
    x = concatenate([x, x2])
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 5
    x = UpSampling2D(size=(2, 2))(x)
    x = concatenate([x, x1])
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Upsample to original size
    x = UpSampling2D(size=(2, 2))(x)

    # Final layer to produce the segmentation mask
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

def RUnet_Segmentation_Model(input_shape=(None, 256, 256, 3), num_classes=2, activation='relu', dropout_rate=0.5):

    def encoder_block(inp, filters, pool=True):
        conv = Conv2D(filters, (3, 3), activation=activation, padding='same')(inp)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters, (3, 3), activation=activation, padding='same')(conv)
        conv = BatchNormalization()(conv)
        if pool:
            pool = MaxPooling2D((2, 2))(conv)
            return pool, conv
        else:
            return conv

    def decoder_block(inp, concat_inp, filters):
        upsample = UpSampling2D((2, 2))(inp)
        concat = Concatenate(axis=-1)([upsample, concat_inp])
        conv = Conv2D(filters, (3, 3), activation=activation, padding='same')(concat)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters, (3, 3), activation=activation, padding='same')(conv)
        conv = BatchNormalization()(conv)
        if dropout_rate:
            conv = Dropout(dropout_rate)(conv)
        return conv

    input_layer = Input(shape=input_shape)

    convlstm_out = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=False)(input_layer)

    p1, c1 = encoder_block(convlstm_out, 64)
    p2, c2 = encoder_block(p1, 128)
    p3, c3 = encoder_block(p2, 256)
    p4, c4 = encoder_block(p3, 512)
    c5 = encoder_block(p4, 1024, pool=False)

    d1 = decoder_block(c5, c4, 512)
    d2 = decoder_block(d1, c3, 256)
    d3 = decoder_block(d2, c2, 128)
    d4 = decoder_block(d3, c1, 64)

    segment_out = Conv2D(num_classes, (1, 1), activation='softmax')(d4)

    model = Model(inputs=input_layer, outputs=segment_out)
    return model


def TemporalUNet_Segmentation_Model(input_shape=(None, 256, 256, 3),
                                    num_classes=2):  # None indicates variable sequence length
    # U-Net encoder block
    def encoder_block(inp, filters, pool=True):
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(inp)
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
        if pool:
            pool = MaxPooling2D((2, 2))(conv)
            return pool, conv
        else:
            return conv

    # U-Net decoder block
    def decoder_block(inp, concat_inp, filters):
        upsample = UpSampling2D((2, 2))(inp)
        concat = Concatenate()([upsample, concat_inp])
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(concat)
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv)
        return conv

    # Starting with the recurrent layer for the input sequence
    input_layer = Input(shape=input_shape)

    # Flatten the spatial dimensions and retain time dimension
    rnn_input = TimeDistributed(Reshape((-1,)))(input_layer)

    # LSTM layer for sequence
    lstm_out = LSTM(input_shape[1] * input_shape[2] * input_shape[3], return_sequences=True)(rnn_input)

    # Reshape LSTM output back to image format
    lstm_reshaped = Reshape((-1, input_shape[1], input_shape[2], input_shape[3]))(
        lstm_out)  # -1 for variable sequence length

    # U-Net architecture begins here
    p1, c1 = encoder_block(lstm_reshaped, 64)
    p2, c2 = encoder_block(p1, 128)
    p3, c3 = encoder_block(p2, 256)
    p4, c4 = encoder_block(p3, 512)

    # Middle part of U-Net
    c5 = encoder_block(p4, 1024, pool=False)

    # Decoder of U-Net
    d1 = decoder_block(c5, c4, 512)
    d2 = decoder_block(d1, c3, 256)
    d3 = decoder_block(d2, c2, 128)
    d4 = decoder_block(d3, c1, 64)

    # Final output
    segment_out = Conv2D(num_classes, (1, 1), activation='softmax')(d4)

    model = Model(inputs=input_layer, outputs=segment_out)
    return model

def RCNN_Segmentation_Model(input_shape=(10, 256, 256, 3), num_classes=2):
    # CNN Sub-model
    cnn_input = Input(shape=input_shape[1:])  # (256, 256, 3)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(cnn_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    cnn_model = Model(cnn_input, x)

    # Main model
    input_layer = Input(shape=input_shape)  # (5, 256, 256, 3)

    # Apply the cnn_model on each time step
    cnn_out = TimeDistributed(cnn_model)(input_layer)

    # Flatten the spatial dimensions while retaining the sequential dimension
    cnn_out_flat = TimeDistributed(Flatten())(cnn_out)

    # After the LSTM layer
    lstm_out = LSTM(256, return_sequences=False)(cnn_out_flat)  # Do not return sequences

    # Dense layer to adjust the output size to (64, 64, 4)
    dense_out = Dense(64 * 64 * 4)(lstm_out)

    # Reshape LSTM output to fit the segmentation layer
    reshape_out = Reshape((64, 64, 4))(dense_out)

    # Upsample to match target resolution
    upsample_out = UpSampling2D(size=(4, 4))(reshape_out)  # 64*4 = 256

    # Segmentation layer
    segment_out = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(upsample_out)

    # Create and compile the model
    model = Model(inputs=input_layer, outputs=segment_out)
    return model


def cnn_timedistributed_lstm2d(sequence_length, input_size):


    input_shape = (sequence_length, input_size, input_size, 3)  # Input shape for sequence

    inputs = Input(shape=input_shape)

    # Reshape input to be compatible with TimeDistributed
    reshaped_input = Reshape((sequence_length, input_size, input_size, 3))(inputs)

    # Pre-trained MobileNetV2 without top and without weights
    mobilenet_encoder = MobileNetV2(include_top=False, weights=None, input_shape=(input_size, input_size, 3))

    # TimeDistributed wrapper to apply CNN across time dimension of sequences
    encoder_output = TimeDistributed(mobilenet_encoder)(reshaped_input)

    # ConvLSTM2D layer with return_sequences=True to output a sequence of feature maps
    convlstm = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=True)(encoder_output)

    # Upsampling and convolution layers applied to each time step
    upsampled = TimeDistributed(UpSampling2D(size=(2, 2)))(convlstm)
    conv1 = TimeDistributed(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))(upsampled)
    upsampled = TimeDistributed(UpSampling2D(size=(2, 2)))(conv1)
    conv2 = TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))(upsampled)

    # Output layer with sigmoid activation for binary segmentation
    outputs = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same'))(conv2)

    # Reshape output to match the label shape (sequence_length, 256, 256)
    reshaped_output = Reshape((sequence_length, input_size, input_size))(outputs)

    model = Model(inputs=inputs, outputs=reshaped_output)

    return model