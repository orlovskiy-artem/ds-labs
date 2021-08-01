from tensorflow import keras as ks

class EncoderBlock(ks.layers.Layer):
    def __init__(self,
                 num_filters,
                 conv_kernel_size=(3,3),
                 activation="relu",
                 padding_conv="same",
                 pool_size = (2,2),
                 padding_pool="same",
                 use_batchnorm=False):
        super(EncoderBlock, self).__init__()
        #parameters
        self.num_filters = num_filters
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation
        self.padding_conv = padding_conv
        self.pool_size = pool_size
        self.padding_pool = padding_pool
        # By default results are better without batchnorm
        self.use_batchnorm = use_batchnorm
        
        #block
        self._conv2d = ks.layers.Conv2D(self.num_filters,
                             self.conv_kernel_size, 
                             activation=self.activation,
                             padding=self.padding_conv)
        self._max_pool = ks.layers.MaxPooling2D(self.pool_size,
                                   padding=self.padding_pool)
        self._batch_norm = ks.layers.BatchNormalization()
        
    def get_config(self):
        cfg = super().get_config()
        for k,v in vars(self).items():
            if not k.startswith("_"):
                cfg[k]=v
        return cfg    
        
    def call(self,inputs):
        x = inputs 
        x = self._conv2d(x)
        x = self._max_pool(x)
        if self.use_batchnorm:
            x = self._batch_norm(x)
        return x
    
class DecoderBlock(ks.layers.Layer):
    def __init__(self,
                 num_filters,
                 conv_kernel_size=(3,3),
                 activation="relu",
                 padding_conv="same",
                 upsample_size = (2,2),
                 use_batchnorm=False):
        super(DecoderBlock, self).__init__()
        #parameters
        self.num_filters = num_filters
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation
        self.padding_conv = padding_conv
        self.upsample_size = upsample_size
        # By default results are better without it
        self.use_batchnorm = use_batchnorm
        
        #block
        self._upsampling2d = ks.layers.UpSampling2D(self.upsample_size)
        self._conv2d = ks.layers.Conv2D(self.num_filters,
                             self.conv_kernel_size, 
                             activation=self.activation,
                             padding=self.padding_conv)
        self._batch_norm = ks.layers.BatchNormalization()
        
        
    def get_config(self):
        cfg = super().get_config()
        for k,v in vars(self).items():
            if not k.startswith("_"):
                cfg[k]=v
        return cfg    
        
    def call(self,inputs):
        x = inputs 
        x = self._upsampling2d(x)
        x = self._conv2d(x)
        if self.use_batchnorm:
            x = self._batch_norm(x)
        return x