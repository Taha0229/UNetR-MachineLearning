import os
from pathlib import Path
from UNetRMultiClass.entity.config_entity import PrepareModelConfig
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from math import log2


class PrepareModel:
    def __init__(self, config: PrepareModelConfig):
        self.config = config

    def mlp(self, x):
        x = L.Dense(self.config.params_mlp_dim, activation="gelu")(x)
        x = L.Dropout(self.config.params_dropout_rate)(x)
        x = L.Dense(self.config.params_hidden_dim)(x)
        x = L.Dropout(self.config.params_dropout_rate)(x)
        
        return x
    
    def transformer_encoder(self, x):
    
        skip_1 = x
        x = L.LayerNormalization()(x)
        x = L.MultiHeadAttention(num_heads=self.config.params_num_heads, key_dim=self.config.params_hidden_dim)(x,x)
        x = L.Add()([x, skip_1])
        
        skip_2 = x
        x = L.LayerNormalization()(x)
        x = self.mlp(x)
        x = L.Add()([x, skip_2])
        
        return x
    
    def conv_block(self, x, num_filters, kernel_size=3):
        x = L.Conv2D(num_filters, kernel_size=kernel_size, padding="same")(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        
        return x
        
    def deconv_block(self, x, num_filters):
        x = L.Conv2DTranspose(num_filters, kernel_size=2, padding="same", strides=2)(x)
        return x
    
    def get_full_model(self):
        """ inputs """
    
        input_shape = (self.config.params_num_patches, self.config.params_patch_size*self.config.params_patch_size*self.config.params_num_channels)
        inputs = L.Input(input_shape)  ## (None, 256, 768)
        
        """ Patch + Positional Embeddings """
        patch_embed = L.Dense(self.config.params_hidden_dim)(inputs)  ## (None, 256, 768)
        
        positions = tf.range(start=0, limit=self.config.params_num_patches, delta=1) ## (256, )
        
        pos_embed = L.Embedding(input_dim=self.config.params_num_patches, output_dim=self.config.params_hidden_dim)(positions)  ## (256, 768)
        
        x = patch_embed + pos_embed
        
        skip_connection_indexes = [3, 6, 9, 12]
        skip_connections = []
        for i in range(1, self.config.params_num_layers +1, 1):
            x = self.transformer_encoder(x)  ## (None, 256, 768)
            
            if i in skip_connection_indexes:
                skip_connections.append(x)
                
        """ CNN Decoder  """
        
        
        z3, z6, z9, z12 = skip_connections
        
        size = self.config.params_image_size // self.config.params_patch_size
        
        """ Reshaping """
        z0 = L.Reshape((self.config.params_image_size, self.config.params_image_size, self.config.params_num_channels))(inputs)  ## (None, 256, 256, 3)
        
        z3 = L.Reshape((size, size, z3.shape[-1]))(z3)  ## (None, 16, 16, 768)
        z6 = L.Reshape((size, size, z6.shape[-1]))(z6)  ## (None, 16, 16, 768)
        z9 = L.Reshape((size, size, z9.shape[-1]))(z9)  ## (None, 16, 16, 768)
        z12 = L.Reshape((size, size, z12.shape[-1]))(z12)  ## (None, 16, 16, 768)
        
        ## Decoder 1
        x = self.deconv_block(z12, 512)
        
        s = self.deconv_block(z9, 512)
        s = self.conv_block(s, 512)
        
        x = L.Concatenate()([x,s])
        x = self.conv_block(x, 512)
        x = self.conv_block(x, 512)
        
        ## Decoder 2
        x = self.deconv_block(x, 256)
        
        s = self.deconv_block(z6, 256)
        s = self.conv_block(s, 256)
        s = self.deconv_block(s, 256)
        s = self.conv_block(s, 256)
        
        x = L.Concatenate()([x, s])
        x = self.conv_block(x, 256)
        x = self.conv_block(x, 256)
        
        ## Decoder 3
        x = self.deconv_block(x, 128)
        
        s = self.deconv_block(z3, 128)
        s = self.conv_block(s, 128)
        s = self.deconv_block(s, 128)
        s = self.conv_block(s, 128)
        s = self.deconv_block(s, 128)
        s = self.conv_block(s, 128)
        
        x = L.Concatenate()([x, s])
        x = self.conv_block(x, 128)
        x = self.conv_block(x, 128)
        
        ## Decoder 4
        x = self.deconv_block(x, 64)
        
        s = self.conv_block(z0, 64)
        s = self.conv_block(s, 64)
        
        x = L.Concatenate()([x, s])
        x = self.conv_block(x, 64)
        x = self.conv_block(x, 64)
        
        """ Output """
        outputs = L.Conv2D(self.config.params_num_classes, kernel_size=1, padding="same", activation="sigmoid")(x) ## 1 -> binary segmentation and hence the sigmoid fxn, can change for multi-class
        full_model = Model(inputs, outputs, name="UNETR_2D")
        full_model.compile(loss="categorical_crossentropy", optimizer=SGD(self.config.params_learning_rate))
        
        
        full_model.summary()
        self.save_model(path=self.config.model_path, model=full_model)
        
        return full_model

    def get_lite_model(self):
        """ Inputs """
        input_shape = (self.config.params_num_patches, self.config.params_patch_size * self.config.params_patch_size * self.config.params_num_channels)
        inputs = L.Input(input_shape) ## (None, 256, 3072)
        # print(inputs.shape)

        """ Patch + Position Embeddings """
        patch_embed = L.Dense(self.config.params_hidden_dim)(inputs) ## (None, 256, 768)

        positions = tf.range(start=0, limit=self.config.params_num_patches, delta=1) ## (256,)
        pos_embed = L.Embedding(input_dim=self.config.params_num_patches, output_dim=self.config.params_hidden_dim)(positions) ## (256, 768)
        x = patch_embed + pos_embed ## (None, 256, 768)

        """ Transformer Encoder """
        skip_connection_index = [3, 6, 9, 12]
        skip_connections = []

        for i in range(1, self.config.params_num_layers+1, 1):
            x = self.transformer_encoder(x)

            if i in skip_connection_index:
                skip_connections.append(x)

        """ CNN Decoder """
        z3, z6, z9, z12 = skip_connections

        ## Reshaping
        z0 = L.Reshape((self.config.params_image_size, self.config.params_image_size, self.config.params_num_channels))(inputs)

        shape = (
            self.config.params_image_size//self.config.params_patch_size,
            self.config.params_image_size//self.config.params_patch_size,
            self.config.params_hidden_dim
        )
        z3 = L.Reshape(shape)(z3)
        z6 = L.Reshape(shape)(z6)
        z9 = L.Reshape(shape)(z9)
        z12 = L.Reshape(shape)(z12)

        ## Additional layers for managing different patch sizes
        total_upscale_factor = int(log2(self.config.params_patch_size))
        upscale = total_upscale_factor - 4

        if upscale >= 2: ## Patch size 16 or greater
            z3 = self.deconv_block(z3, z3.shape[-1], strides=2**upscale)
            z6 = self.deconv_block(z6, z6.shape[-1], strides=2**upscale)
            z9 = self.deconv_block(z9, z9.shape[-1], strides=2**upscale)
            z12 = self.deconv_block(z12, z12.shape[-1], strides=2**upscale)
            # print(z3.shape, z6.shape, z9.shape, z12.shape)

        if upscale < 0: ## Patch size less than 16
            p = 2**abs(upscale)
            z3 = L.MaxPool2D((p, p))(z3)
            z6 = L.MaxPool2D((p, p))(z6)
            z9 = L.MaxPool2D((p, p))(z9)
            z12 = L.MaxPool2D((p, p))(z12)

        ## Decoder 1
        x = self.deconv_block(z12, 128)

        s = self.deconv_block(z9, 128)
        s = self.conv_block(s, 128)

        x = L.Concatenate()([x, s])

        x = self.conv_block(x, 128)
        x = self.conv_block(x, 128)

        ## Decoder 2
        x = self.deconv_block(x, 64)

        s = self.deconv_block(z6, 64)
        s = self.conv_block(s, 64)
        s = self.deconv_block(s, 64)
        s = self.conv_block(s, 64)

        x = L.Concatenate()([x, s])
        x = self.conv_block(x, 64)
        x = self.conv_block(x, 64)

        ## Decoder 3
        x = self.deconv_block(x, 32)

        s = self.deconv_block(z3, 32)
        s = self.conv_block(s, 32)
        s = self.deconv_block(s, 32)
        s = self.conv_block(s, 32)
        s = self.deconv_block(s, 32)
        s = self.conv_block(s, 32)

        x = L.Concatenate()([x, s])
        x = self.conv_block(x, 32)
        x = self.conv_block(x, 32)

        ## Decoder 4
        x = self.deconv_block(x, 16)

        s = self.conv_block(z0, 16)
        s = self.conv_block(s, 16)

        x = L.Concatenate()([x, s])
        x = self.conv_block(x, 16)
        x = self.conv_block(x, 16)

        """ Output """
        outputs = L.Conv2D(self.config.params_num_classes, kernel_size=1, padding="same", activation="sigmoid")(x)

        lite_model = Model(inputs, outputs, name="UNETR_2D_lite")
        lite_model.compile(loss="categorical_crossentropy", optimizer=SGD(self.config.params_learning_rate))
        
        
        lite_model.summary()
        self.save_model(path=self.config.model_path, model=lite_model)
        
        return lite_model
        
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
