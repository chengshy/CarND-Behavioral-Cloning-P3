import csv
import cv2
import numpy as np
import sklearn

lines = []
with open('./recorded_data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print('Dataset has %d data' % len(lines))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size = 0.2)
print('Original Train data size is %d' % len(train_samples))
print('Original Validation data size is %d' % len(validation_samples))

def generator(samples, batch_size = 1024):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]

            center_images = []
            center_measurements = []
            left_images = []
            left_measurements = []
            right_images = []
            right_measurements = []
            for batch_sample in batch_samples:
                center_path = batch_sample[0]
                left_path = batch_sample[1]
                right_path = batch_sample[2]
                filename_center = center_path.split('/')[-1]
                filename_left = left_path.split('/')[-1]
                filename_right = right_path.split('/')[-1]
                center_path = './recorded_data/IMG/' + filename_center
                left_path = './recorded_data/IMG/' + filename_left
                right_path = './recorded_data/IMG/' + filename_right
                image_center = cv2.imread(center_path)
                image_left = cv2.imread(left_path)
                image_right = cv2.imread(right_path)

i               # Convert to YUV
                image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2YUV)
                image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2YUV)
                image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2YUV)
                center_images.append(image_center)
                left_images.append(image_left)
                right_images.append(image_right)
                center_measurement = float(batch_sample[3])
            
                correction = 0.1 + center_measurement * center_measurement / 0.45;
                left_measurement = center_measurement + correction
                right_measurement = center_measurement - correction
            
                center_measurements.append(center_measurement)
                left_measurements.append(left_measurement)
                right_measurements.append(right_measurement)
            
            assert(len(center_images) == len(center_measurements))
            assert(len(left_images) == len(left_measurements))
            assert(len(right_images) == len(right_measurements))
            
            images = center_images + left_images + right_images
            measurements = center_measurements + left_measurements + right_measurements
            
            assert(len(images) == len(measurements))

            augmented_images = []
            augmented_measurements = []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement * -1.0)
            
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

BATCH_SIZE  = 128
train_generator = generator(train_samples, batch_size = BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size = BATCH_SIZE)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#Crop
model.add(Cropping2D(cropping = ((70, 25), (0, 0)), input_shape = (160, 320, 3)))
#Normalize 
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
#Nvidia NN
model.add(Convolution2D(24,5,5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch = 6 * len(train_samples),
    validation_data = validation_generator, 
    nb_val_samples = len(validation_samples), nb_epoch = 5,
    verbose = 1)

model.save('model.h5')
