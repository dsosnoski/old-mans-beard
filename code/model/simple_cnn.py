
import json
import numpy as np

from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers as layers

from model.sample_sequence import SampleStore, SampleSequence

BASE_PATH = '/home/dennis/projects/wcc/images'
VALIDATE_FRACTION = .15
DATA_FILE = '/home/dennis/projects/wcc/data.npy'
META_FILE = '/home/dennis/projects/wcc/data.json'
BATCH_SIZE = 32
EPOCHS = 20
INPUT_SHAPE = (800, 800, 3)
OUTPUT_SCALE = 2
OUTPUT_SHAPE = ((INPUT_SHAPE[0] // OUTPUT_SCALE), (INPUT_SHAPE[1] // OUTPUT_SCALE))

with open(DATA_FILE, 'rb') as f:
    use_samples = np.load(f)
    use_probs = np.load(f)
    test_samples = np.load(f)
    test_probs = np.load(f)
with open(META_FILE, 'r') as f:
    metadata = json.load(f)

actuals = (use_probs * 255).astype(np.uint8)
# use_trues = np.sum(actuals, axis=(1,2))
# use_order = np.argsort(use_trues)
validate_count = int(len(use_samples) * VALIDATE_FRACTION)
while True:
    permute = np.random.default_rng().permutation(len(use_samples))
    training_samples = use_samples[permute][:-validate_count]
    training_actuals = actuals[permute][:-validate_count]
    validation_samples = use_samples[permute][-validate_count:]
    validation_actuals = actuals[permute][-validate_count:]
    if np.sum(validation_actuals) > 0:
        break
training_store = SampleStore(training_samples, INPUT_SHAPE, training_actuals, OUTPUT_SHAPE, 10.0, 30, .5, True)
training_sequence = SampleSequence(training_store, BATCH_SIZE)
validation_store = SampleStore(validation_samples, INPUT_SHAPE, validation_actuals, OUTPUT_SHAPE)
validation_sequence = SampleSequence(validation_store, BATCH_SIZE)

# create model
model = Sequential()
model.add(layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=INPUT_SHAPE))
model.add(layers.Conv2D(32, 3, strides=(2,2), padding='same', activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(1, 1, padding='same', activation='sigmoid'))

# take a look at the model summary
model.summary()

# compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(training_sequence, batch_size=BATCH_SIZE, validation_data=validation_sequence, validation_batch_size=validate_count, epochs=EPOCHS, verbose=1)

# evaluate the result
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])


