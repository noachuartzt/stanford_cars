import pickle
import numpy as np

from settings import settings

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, Dropout, AveragePooling2D

class LeNet:
    
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

        # Layer 1
        x = Conv2D(filters=6, kernel_size=5, strides=1, activation='tanh')(inputs)
        x = AveragePooling2D(strides=2)(x)

        # Layer 2
        x = Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh')(x)
        x = AveragePooling2D(strides=2)(x)

        # Layer 3
        x = Conv2D(filters=120, kernel_size=5, strides=1, activation='tanh')(x)
        x = Flatten()(x)

        # Fully Connected Layers
        x = Dense(units=84, activation='tanh')(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=output)
        
        return model
    
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy']):
        """Compiles the model."""

        return self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        

    def train(self, train_generator, validation_generator, epochs = 100, patience = 5):
        """Trains the model."""
        
        history_path = settings.lenet_model + 'history.pkl'
        
        try:
            # Open model
            self.model = load_model(settings.lenet_model[:-1])
            print('Model loaded')            
            
            # Open history
            with open(history_path, 'rb') as file:
                history = pickle.load(file)
        
            print('History loaded')
            
        except:
            print('Model not found')
            print('Training model...')
            
            # Train model
            history = self.model.fit(train_generator,
                                     validation_data=validation_generator, 
                                     epochs=epochs, 
                                     callbacks=[EarlyStopping(patience=patience)])
            
            # Save model
            self.model.save(settings.lenet_model)
            
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

    