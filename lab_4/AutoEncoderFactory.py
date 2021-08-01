from tensorflow import keras as ks
from AutoEncoderBlocks import EncoderBlock,DecoderBlock
from tqdm import tqdm 
from copy import deepcopy

class IncrementalAutoencoderFactory:
    """Factory class that generates autoencoders one by one, 
       setting trained layers of previous encoders to the new one
       
       Autoencoders use EncoderBlocks and DecoderBlocks, which are convolutional-based.
       
       To work with it, use it like this
       >>> autoencoder_factory = IncrementalAutoencoderFactory([16,8,4],shape=(256,256,3))
       >>> autoencoder_factory.build()
       >>> autoencoder_factory.fit_factory(train_data_generator,validation_data_generator,
       ...      epochs,callbacks = your_callbacks,**your_fit_parameters)
       >>> autoencoder_factory.predict(test_data,3)
     """
    
    def __init__(self,
                 filters_sizes,
                 data_shape,
                 noisy=True,
                 gaussian_noise_std = 0.2,
                 metrics = ["mse",ks.metrics.kl_divergence],
                 loss = "binary_crossentropy",
                 optimizer = "adam",
                 encoder_block_params = {},
                 decoder_block_params = {}):
        self.filters_sizes = filters_sizes
        self.encoders = []
        self.decoders = []
        self.autoencoders = []
        self.noisy = noisy
        self.gaussian_noise_std= gaussian_noise_std
        self.data_shape = data_shape
        self.metrics = metrics
        self.loss = loss
        self.optimizer = optimizer
        self.encoder_block_params = encoder_block_params
        self.decoder_block_params = decoder_block_params
        
    def build(self):
        """ This method builds first encoder and decoder. Should be called once after initialization"""
        encoder = ks.Sequential(name = "encoder")
        encoder.add(ks.layers.Input(shape = self.data_shape))
        encoder.add(EncoderBlock(self.filters_sizes[0],**self.encoder_block_params))
        
        decoder = ks.Sequential(name = "decoder")
        decoder.add(ks.layers.Input(shape = encoder.layers[-1].output.shape[1:]))
        decoder.add(DecoderBlock(self.data_shape[2],**self.decoder_block_params))
        
        if self.noisy:
            autoencoder = ks.Sequential([
                ks.layers.GaussianNoise(self.gaussian_noise_std),encoder,decoder
            ],name="denoising_autoencoder")
        else:
            autoencoder = ks.Sequential([
                encoder,decoder
            ],name="autoencoder")
        
        autoencoder.compile(self.optimizer,loss=self.loss,metrics=self.metrics)
        self.encoders.append(encoder)
        self.decoders.append(decoder)
        self.autoencoders.append(autoencoder)
    
    def set_non_trainable(self,model):
        """Sets all layers to not trainable"""
        for i in range(len(model.layers)):
            model.layers[i].trainable=False
        return model
    
    def deepen_autoencoder(self,
                           filter_size=None):
        """Gets last_trained autoencoder. Sets all it's layers to non-trainable state 
           and adds new EncoderBlock and DecoderBlock to the center of the model. 
        :param filter_size: number of feature-maps in new EncoderBlock layer. If it's None than filters_sizes 
                            from initializer are used, otherwise it can be set to new number if last autoencoder
                            is not deep enough. In this case, method adds new EncoderBlock layer to the end of the encoder
                            and new DecoderBlock layer at the beginning of the decoder.
        """
        depth = len(self.encoders)
        if filter_size is None:
            filter_size = self.filters_sizes[depth]
        else:
            self.filters_sizes.append(filter_size)
        last_encoder = deepcopy(self.encoders[-1])
        last_decoder = deepcopy(self.decoders[-1])
        
        last_encoder = self.set_non_trainable(last_encoder)
        last_decoder = self.set_non_trainable(last_decoder)
        
        decoder_new_block_filter_num = last_encoder.layers[-1].output.shape[-1]
        
        inputs = ks.layers.Input(shape=self.data_shape)
        x = inputs
        for i in range(len(last_encoder.layers)):
            if type(last_encoder.layers[i])==ks.layers.InputLayer:
                continue
            x = last_encoder.layers[i](x)
        encoded = EncoderBlock(filter_size,**self.encoder_block_params)(x)
        encoder = ks.Model(inputs,encoded)

        newInput = ks.layers.Input(shape=encoder.layers[-1].output.shape[1:])  
        x = DecoderBlock(decoder_new_block_filter_num,**self.decoder_block_params)(newInput)
        for i in range(len(last_decoder.layers)):
            if type(last_decoder.layers[i])==ks.layers.InputLayer:
                continue
            x = last_decoder.layers[i](x)
        outputs = x
        decoder = ks.Model(newInput,outputs)
        autoencoder = ks.Sequential([
            ks.layers.GaussianNoise(self.gaussian_noise_std),encoder,decoder
        ],name="denoising_autoencoder")
        autoencoder.compile(self.optimizer,loss=self.loss,metrics=self.metrics)
        self.encoders.append(encoder)
        self.decoders.append(decoder)
        self.autoencoders.append(autoencoder)
        
    def fit_model(self,
            train_data_generator,
            validation_data_generator,
            epochs,
            callbacks = [],
            **fit_parameters):
        """Only default parameters are tested. Please ask if it's necessary to implement a general case"""
        autoencoder = self.autoencoders[-1]
        autoencoder.fit(train_data_generator,
                        epochs=epochs,
                        validation_data=validation_data_generator,
                        callbacks=callbacks,**fit_parameters)
    def fit_factory(self,
            train_data_generator,
            validation_data_generator,
            epochs,
            callbacks = [],
            **fit_parameters):
        """Method to train all autoencoders in incremental way.
           New layers will be trained, but previous ones will be set to non-trainable.
        """
        # Fit first autoencoder after build
        self.fit_model(train_data_generator,
                       validation_data_generator,
                       epochs,callbacks,**fit_parameters)
            
        # first layer is already created using build method
        for _ in tqdm(range(len(self.filters_sizes)-1)):
            self.deepen_autoencoder()
            self.fit_model(train_data_generator,
                       validation_data_generator,
                       epochs,callbacks,**fit_parameters)
            
            
    def get_encoder_decoder(self, depth):
        """
        :param depth: depth of model, meant number of blocks in encoder or decoder"""
        return self.encoders[depth],self.decoders[depth]
           
    def encode(self, data, depth):
        return self.encoders[depth](data)
        
    def decode(self, data_encoded, depth):
        return  self.decoders[depth](data_encoded)
      
    def predict(self,data,depth):
        encoded = self.encode(data,depth)
        decoded = self.decode(encoded,depth)
        return decoded