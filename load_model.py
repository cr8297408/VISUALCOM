from keras.models import load_model;
from keras.preprocessing.image   import ImageDataGenerator
import numpy as np

model = load_model('model_first.h5');

model.summary();

## data de validación
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=(0.3),
        zoom_range=(0.3),
        width_shift_range=(0.2),
        height_shift_range=(0.2),
        validation_split = 0.2,
        brightness_range=(0.05,0.85),
        horizontal_flip=False,
        rotation_range=20
)

valid_generator = train_datagen.flow_from_directory(
        './train',
        class_mode='categorical',
        shuffle=True,
        target_size=(28, 28),
        color_mode = 'rgb', 
        subset='validation',
        batch_size=32)

print(valid_generator)

predict = model.predict(valid_generator)

print(predict);

# Obtener la clase con la mayor probabilidad
clase_predicha = np.argmax(predict, axis=1)

# Imprimir la clase predicha
print(clase_predicha)