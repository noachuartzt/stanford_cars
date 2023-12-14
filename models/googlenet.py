import pickle
import numpy as np

from settings import settings

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.layers import concatenate, AveragePooling2D

class GoogleNet:
    
    def __init__(self, input_shape, num_classes, see_summary = False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.compile_model()
        
        if see_summary:
            print(self.model.summary())
        
    def inception_module(self, layer, f1, f3_reduce, f3, f5_reduce, f5, pool_proj, name=None):
        
        conv1 = Conv2D(filters=f1, kernel_size=1, activation='relu', padding='same')(layer)
        
        conv3 = Conv2D(filters=f3_reduce, kernel_size=1, activation='relu', padding='same')(layer)
        conv3 = Conv2D(filters=f3, kernel_size=3, activation='relu', padding='same')(conv3)
        
        conv5 = Conv2D(filters=f5_reduce, kernel_size=1, activation='relu', padding='same')(layer)
        conv5 = Conv2D(filters=f5, kernel_size=5, activation='relu', padding='same')(conv5)
        
        pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(layer)
        pool = Conv2D(filters=pool_proj, kernel_size=1, activation='relu', padding='same')(pool)
        
        output = concatenate([conv1, conv3, conv5, pool], axis=-1, name=name)
        
        return output

   
    def build_model(self):
        """Builds a convolutional neural network model."""

        inputs = Input(shape=self.input_shape)

        # Layer 1
        x = Conv2D(filters=64, kernel_size=7, strides=2, activation='relu', padding='same', name='conv_1')(inputs)
        x = MaxPool2D(pool_size=3, strides=2, padding='same', name='maxpool_1')(x)
        
        # Layer 2
        x = Conv2D(filters=64, kernel_size=1, activation='relu', padding='same', name='conv_2')(x)
        x = Conv2D(filters=192, kernel_size=3, activation='relu', padding='same', name='conv_3')(x)
        x = MaxPool2D(pool_size=3, strides=2, padding='same', name='maxpool_2')(x)
        
        # Layer 3
        x = self.inception_module(x, f1=64, f3_reduce=96, f3=128, f5_reduce=16, f5=32, pool_proj=32, name='inception_3a')
        x = self.inception_module(x, f1=128, f3_reduce=128, f3=192, f5_reduce=32, f5=96, pool_proj=64, name='inception_3b')
        x = MaxPool2D(pool_size=3, strides=2, padding='same', name='maxpool_3')(x)
        
        # Layer 4
        x = self.inception_module(x, f1=192, f3_reduce=96, f3=208, f5_reduce=16, f5=48, pool_proj=64, name='inception_4a')
        x = self.inception_module(x, f1=160, f3_reduce=112, f3=224, f5_reduce=24, f5=64, pool_proj=64, name='inception_4b')
        x = self.inception_module(x, f1=128, f3_reduce=128, f3=256, f5_reduce=24, f5=64, pool_proj=64, name='inception_4c')
        x = self.inception_module(x, f1=112, f3_reduce=144, f3=288, f5_reduce=32, f5=64, pool_proj=64, name='inception_4d')
        
        # Layer 5
        x = self.inception_module(x, f1=256, f3_reduce=160, f3=320, f5_reduce=32, f5=128, pool_proj=128, name='inception_4e')
        x = MaxPool2D(pool_size=3, strides=2, padding='same', name='maxpool_4')(x)
        
        # Layer 6
        x = self.inception_module(x, f1=256, f3_reduce=160, f3=320, f5_reduce=32, f5=128, pool_proj=128, name='inception_5a')
        x = self.inception_module(x, f1=384, f3_reduce=192, f3=384, f5_reduce=48, f5=128, pool_proj=128, name='inception_5b')
        x = AveragePooling2D(pool_size=7, strides=1, padding='valid', name='avgpool_1')(x)
        
        # Layer 7
        x = Dropout(0.4)(x)
        x = Flatten()(x)
        x = Dense(units=1000, activation='relu')(x)
        output = Dense(units=self.num_classes, activation='softmax', name='fc')(x)
        
        model = Model(inputs=inputs, outputs=output)
        return model
    
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy']):
        """Compiles the model."""

        return self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        

    def train(self, train_generator, validation_generator, 
              epochs = 100, batch_size = 32, verbose = 0, patience = 5):        
        
        """Trains the model."""
        
        history_path = settings.googlenet_model + '/history.pkl'
        
        try:
            # Open model
            self.model = load_model(settings.googlenet_model)
            print('Model loaded')
                        
            # Open history
            with open(history_path, 'rb') as file:
                history = pickle.load(file)
        
            print('History loaded')
            
        except:
            print('Model not found')
            print('Training model...')
            
            # Create callbacks
            checkpoint = ModelCheckpoint(settings.googlenet_model, save_best_only=True, monitor='val_loss', mode='min',verbose=verbose)
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True, verbose=verbose)
            
            # Train model
            history = self.model.fit(train_generator,
                                     validation_data=validation_generator, 
                                     epochs=epochs, 
                                     batch_size=batch_size,
                                     callbacks=[checkpoint, early_stopping])
            
            # Save history
            with open(history_path, 'wb') as file:
                pickle.dump(history, file)
                
            print('Model and history saved')
            
        return history
    
    
    def evaluate(self, test_generator):
        """Evaluates the model."""        
        
        # Evaluate model
        loss, acc = self.model.evaluate(test_generator)
        
        print(f'Loss: {loss:.2f}')
        print(f'Accuracy: {acc*100:.2f}%')
        
        return loss, acc
        
        
    def predict(self, generator):
        """Predicts the class of the input image."""
            
        # Predictions
        return np.argmax(self.model.predict(generator), axis=1)

    