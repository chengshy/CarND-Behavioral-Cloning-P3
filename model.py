import csv
import cv2
import numpy as np
import sklearn

def get_filenames(path_prefix, log_name, random_drop_small_angle = True):
    lines = []
    with open(path_prefix + log_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if abs(float(line[3])) < 0.05:
                if np.random.randint(100) < 65:
                    continue
            lines.append(line)
    print('%s Dataset has %d data' % (log_name, len(lines)) )
    return lines

path_prefix = './all_data/'
lines_turn = get_filenames(path_prefix, 'driving_log_turn.csv')
lines_first = get_filenames(path_prefix, 'driving_log_first_track.csv')
lines_second = get_filenames(path_prefix, 'driving_log_second_track.csv')
lines_recovery = get_filenames(path_prefix, 'driving_log_recovery.csv')
lines_carnd = get_filenames(path_prefix, 'driving_log_carnd.csv')

lines = []
lines += lines_turn + lines_carnd + lines_recovery + lines_first + lines_second
print('Dataset has %d data' % len(lines))

from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(lines, test_size = 0.2)
print('Original Train data size is %d' % len(train_samples))
print('Original Validation data size is %d' % len(validation_samples))

def generator(samples, path_prefix, batch_size = 1024, use_left_right = False):
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
                center_path = path_prefix + 'IMG/' + filename_center
                image_center = cv2.imread(center_path)
                image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2YUV)
                center_images.append(image_center)
                center_measurement = float(batch_sample[3])
                center_measurements.append(center_measurement)

                if use_left_right:
                    left_path = path_prefix + 'IMG/' + filename_left
                    right_path = path_prefix + 'IMG/' + filename_right
                    image_left = cv2.imread(left_path)
                    image_right = cv2.imread(right_path)
                    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2YUV)
                    image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2YUV)
                    left_images.append(image_left)
                    right_images.append(image_right)
                    # correction = 0.175 + center_measurement * center_measurement / 0.45;
                    correction = 0.23;
                    left_measurement = center_measurement + correction
                    right_measurement = center_measurement - correction
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

BATCH_SIZE  = 64
use_left_right = True
train_generator = generator(train_samples, path_prefix, batch_size = BATCH_SIZE, use_left_right = use_left_right)
validation_generator = generator(validation_samples, path_prefix, batch_size = BATCH_SIZE, use_left_right = use_left_right)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

model = Sequential()
#Crop
model.add(Cropping2D(cropping = ((55, 25), (0, 0)), input_shape = (160, 320, 3)))
#Normalize 
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
#Nvidia NN
model.add(Convolution2D(24,5,5, subsample = (2,2), activation = 'elu'))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation = 'elu'))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation = 'elu'))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(64,3,3, activation = 'elu'))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(64,3,3, activation = 'elu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation = 'elu'))
model.add(Dense(50, activation = 'elu'))
model.add(Dense(10, activation = 'elu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr = 0.001))

checkpointer = ModelCheckpoint(filepath="/tmp/weights.h5", verbose=1, save_best_only=True)

earlystop = EarlyStopping(patience = 1)

samples_num = 6 * len(train_samples) if use_left_right else 2 * len(train_samples)
validation_num = 6 * len(validation_samples) if use_left_right else 2 * len(validation_samples)

model.fit_generator(train_generator, samples_per_epoch = samples_num,
    validation_data = validation_generator, 
    nb_val_samples = validation_num, nb_epoch = 50,
    verbose = 1,
    callbacks = [checkpointer, earlystop])

model.save('model.h5')
