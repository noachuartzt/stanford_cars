import pickle
import numpy as np

from settings import settings

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Dropout

class AlexNet:
    
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
        x = Conv2D(filters=96, kernel_size=11, strides=4, padding='valid', activation='relu')(inputs)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        # Layer 2
        x = Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        # Layer 3
        x = Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu')(x)

        # Layer 4 
        x = Conv2D(filters=384, kernel_size=3, strides=1, padding='same', activation='relu')(x)

        # Layer 5
        x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        # Flatten the CNN output to feed it with fully connected layers
        print(x.shape)
        x = Flatten()(x)
        print(x.shape)

        # Layer 6
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)

        # Layer 7
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=output)
        
        return model
    
    
    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy']):
        """Compiles the model."""

        return self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        

    def train(self, train_generator, validation_generator, epochs = 100, patience = 5, verbose = 1):
        """Trains the model."""
        
        history_path = settings.alexnet_model + 'history.pkl'
        
        try:
            # Open model
            self.model = load_model(settings.alexnet_model[:-1])
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
                                     callbacks=[EarlyStopping(patience=patience, verbose=verbose)])
            
            # Save model
            self.model.save(settings.alexnet_model)
            
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

    