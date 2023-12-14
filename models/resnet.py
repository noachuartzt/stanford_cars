import pickle
import numpy as np

from settings import settings

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.layers import concatenate, AveragePooling2D

class ResNet:
    
    def __init__(self, input_shape, num_classes, see_summary = False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.compile_model()
        
        if see_summary:
            print(self.model.summary())
        

    def build_model(self):
        """Builds a convolutional neural network model."""

        inputs = Input(shape=self.input_shape)

        x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        
        x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        
        x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        
        x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        
        x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
        x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        
        # Layer 6
        x = Flatten()(x)
        x = Dense(units=4096, activation='relu')(x)
        x = Dense(units=4096, activation='relu')(x)
        output = Dense(units=self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=output)
        return model
    
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy']):
        """Compiles the model."""

        return self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        

    def train(self, train_generator, validation_generator, 
              epochs = 100, batch_size = 32, verbose = 0, patience = 5):        
        
        """Trains the model."""
    
        history_path = settings.resnet_model + '/history.pkl'
    
        try:
            # Open model
            self.model = load_model(settings.resnet_model)
            print('Model loaded')
            
            # Open history
            with open(history_path, 'rb') as file:
                history = pickle.load(file)
        
            print('History loaded')
            
        except:
            print('Model not found')
            print('Training model...')
            
            # Create callbacks
            checkpoint = ModelCheckpoint(settings.resnet_model, save_best_only=True, monitor='val_loss', mode='min',verbose=verbose)
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

    