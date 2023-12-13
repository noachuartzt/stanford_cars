import os
from functools import lru_cache

class Settings():
    """ Settings."""
    
    data : str = "data/"
    
    # Data Train and Test paths
    data_train : str = os.path.join(data, "anno_train.csv")
    data_test : str = os.path.join(data, "anno_test.csv")
    car_model : str = os.path.join(data, "names.csv")
    
    # Images path
    images : str = os.path.join(data, "car_data/car_data/")
    
    # Train and test paths
    train : str = os.path.join(images, "train/")
    test : str = os.path.join(images, "test/")
    
    # Models
    models : str = "models/"
    checkpoints : str = os.path.join(models, "checkpoints/")
    tl_checkpoints : str = os.path.join(models, "transfer-learning_checkpoints/")
    
    # LeNet
    lenet_model : str = os.path.join(checkpoints, "lenet")
    
    # AlexNet 
    alexnet_model : str = os.path.join(checkpoints, "alexnet")
    
    # VGG16
    vgg16_model : str = os.path.join(checkpoints, "vgg16")
    
    # ResNet50
    resnet50_model : str = os.path.join(checkpoints, "resnet50")
    
    # GoogLeNet
    googlenet_model : str = os.path.join(checkpoints, "googlenet")
    
    # Autoencoder
    autoencoder_model : str = os.path.join(checkpoints, "autoencoder")
    

class Config:
    """Config."""
    env_fiel = ".env"

    
@lru_cache()
def get_settings():
    """Get settings."""
    
    return Settings()

settings = get_settings()