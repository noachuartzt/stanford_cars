import numpy as np
from settings import settings

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class ResNet50_TL:
    
    def __init__(self, input_shape, num_classes, see_summary = False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.compile_model()
        
        if see_summary:
            print(self.model.summary())
            
    def build_model(self):
        """Builds a convolutional neural network model."""

        # Load pre-trained ResNet50 model without top layers
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Freeze base model layers
        for layer in base_model.layers:
            if "BatchNormalization" in layer.__class__.__name__:
                layer.trainable = True
        
        # Add custom layers for classification
        x = GlobalAveragePooling2D()(base_model.output)
        x = BatchNormalization()(x)  

        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)  
        x = BatchNormalization()(x)  

        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)  
        x = BatchNormalization()(x)  

        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)
        
        return model


    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy']):
        """Compiles the model."""

        return self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    
    def train(self, train_generator, validation_generator, 
              epochs = 100, batch_size = 32, verbose = 0, patience = 5):
        
        """Trains the model."""
        
        path = settings.tl_resnet50_model + '_5.ckpt'
        try:
            # Open model
            self.model = load_model(path)
            print('Model loaded')            
            
        except:
            print('Model not found')
            print('Training model')
            
            # Create callbacks 
            checkpoint = ModelCheckpoint(settings.tl_resnet50_model + '_6.ckpt', save_best_only=True, monitor='val_loss', mode='min', verbose=verbose)
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True, verbose=verbose)

            self.model.fit(train_generator, validation_data=validation_generator,
                    epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping])
            
            print('Model trained')
            
            
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

    