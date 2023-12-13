"""This file contains the function to create image generators for the training, validation, and testing data."""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_img_generator(train_path: str, test_path: str, target_size: tuple) -> tuple:
    """Create image generators for the training, validation, and testing data.
    
    Parameters:
        train_path (str): Path to the training data.        
        test_path (str): Path to the testing data.
        target_size (tuple): The target size of the images.
        
    Returns:
        train_generator (generator): Generator for the training data.
        test_generator (generator): Generator for the testing data.
        validation_generator (generator): Generator for the validation data.
    """
    # Preprocess the data
    train_datagen = ImageDataGenerator(rescale=1./255,          # Rescale values from 0-255 to 0-1
                                       rotation_range=20,       # Rotate images by 20 degrees
                                       width_shift_range=0.2,   # Shift images horizontally by 20%
                                       height_shift_range=0.2,  # Shift images vertically by 20%
                                       horizontal_flip=True,    # Flip images horizontally
                                       zoom_range=0.2,          # Zoom in on images by 20%
                                       fill_mode='nearest',     # Fill in missing pixels with the nearest filled value
                                       validation_split=0.2     # Split training data into 80% train and 20% validation
                                       )
                                       
    test_datagen = ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(train_path,
                                                        target_size=target_size,
                                                        class_mode='categorical',
                                                        subset='training')

    validation_generator = train_datagen.flow_from_directory(train_path,
                                                            target_size=target_size,
                                                            class_mode='categorical',
                                                            subset='validation')
    
    test_generator = test_datagen.flow_from_directory(test_path,
                                                    target_size=target_size,
                                                    class_mode='categorical')

    print()
    
    return train_generator, test_generator, validation_generator