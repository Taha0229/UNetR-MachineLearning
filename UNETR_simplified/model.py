import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model

def mlp(x, cf):
    x = L.Dense(cf["mlp_dim"], activation="gelu")(x)
    x = L.Dropout(cf["dropout_rate"])(x)
    x = L.Dense(cf["hidden_dim"])(x)
    x = L.Dropout(cf["dropout_rate"])(x)
    
    return x

def transformer_encoder(x, cf):
    
    skip_1 = x
    x = L.LayerNormalization()(x)
    x = L.MultiHeadAttention(num_heads=cf["num_heads"], key_dim=cf["hidden_dim"])(x,x)
    x = L.Add()([x, skip_1])
    
    skip_2 = x
    x = L.LayerNormalization()(x)
    x = mlp(x, cf)
    x = L.Add()([x, skip_2])
    
    return x
    
def conv_block(x, num_filters, kernel_size=3):
    x = L.Conv2D(num_filters, kernel_size=kernel_size, padding="same")(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
     
    return x
    
def deconv_block(x, num_filters):
    x = L.Conv2DTranspose(num_filters, kernel_size=2, padding="same", strides=2)(x)
    return x
    
def build_unetr_2d(cf):
    """ inputs """
    
    input_shape = (cf["num_patches"], cf["patch_size"]*cf["patch_size"]*cf["num_channels"])
    inputs = L.Input(input_shape)  ## (None, 256, 768)
    
    """ Patch + Positional Embeddings """
    patch_embed = L.Dense(cf["hidden_dim"])(inputs)  ## (None, 256, 768)
    
    positions = tf.range(start=0, limit=cf["num_patches"], delta=1) ## (256, )
    
    pos_embed = L.Embedding(input_dim=cf["num_patches"], output_dim=cf["hidden_dim"])(positions)  ## (256, 768)
    
    x = patch_embed + pos_embed
    
    skip_connection_indexes = [3, 6, 9, 12]
    skip_connections = []
    for i in range(1, cf["num_layers"]+1, 1):
        x = transformer_encoder(x, cf)  ## (None, 256, 768)
        
        if i in skip_connection_indexes:
            skip_connections.append(x)
            
    """ CNN Decoder  """
    
    
    z3, z6, z9, z12 = skip_connections
    
    size = cf["image_size"] // cf["patch_size"]
    
    """ Reshaping """
    z0 = L.Reshape((cf["image_size"], cf["image_size"], cf["num_channels"]))(inputs)  ## (None, 256, 256, 3)
    
    z3 = L.Reshape((size, size, z3.shape[-1]))(z3)  ## (None, 16, 16, 768)
    z6 = L.Reshape((size, size, z6.shape[-1]))(z6)  ## (None, 16, 16, 768)
    z9 = L.Reshape((size, size, z9.shape[-1]))(z9)  ## (None, 16, 16, 768)
    z12 = L.Reshape((size, size, z12.shape[-1]))(z12)  ## (None, 16, 16, 768)
    
    ## Decoder 1
    x = deconv_block(z12, 512)
    
    s = deconv_block(z9, 512)
    s = conv_block(s, 512)
    
    x = L.Concatenate()([x,s])
    x = conv_block(x, 512)
    x = conv_block(x, 512)
    
    ## Decoder 2
    x = deconv_block(x, 256)
    
    s = deconv_block(z6, 256)
    s = conv_block(s, 256)
    s = deconv_block(s, 256)
    s = conv_block(s, 256)
    
    x = L.Concatenate()([x, s])
    x = conv_block(x, 256)
    x = conv_block(x, 256)
    
    ## Decoder 3
    x = deconv_block(x, 128)
    
    s = deconv_block(z3, 128)
    s = conv_block(s, 128)
    s = deconv_block(s, 128)
    s = conv_block(s, 128)
    s = deconv_block(s, 128)
    s = conv_block(s, 128)
    
    x = L.Concatenate()([x, s])
    x = conv_block(x, 128)
    x = conv_block(x, 128)
    
    ## Decoder 4
    x = deconv_block(x, 64)
    
    s = conv_block(z0, 64)
    s = conv_block(s, 64)
    
    x = L.Concatenate()([x, s])
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    
    """ Output """
    outputs = L.Conv2D(1, kernel_size=1, padding="same", activation="sigmoid")(x) ## 1 -> binary segmentation and hence the sigmoid fxn, can change for multi-class
    
    return Model(inputs, outputs, name="UNETR_2D")
    
                     
            
            

if __name__ == "__main__":
    config = {
        "image_size": 256,
        "num_layers": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "dropout_rate": 0.1,
        "num_patches": 256,
        "patch_size": 16,
        "num_channels": 3,
    }
    
    model = build_unetr_2d(cf=config)
    model.summary()
    