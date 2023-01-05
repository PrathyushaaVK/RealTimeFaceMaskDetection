from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.utils import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

#initializing the learning rate, number of epochs and batch size
learningrate = 1e-4
epochs = 20
batchsize = 32

directory = r"C:\Users\shaap\Desktop\prep\Projects\Face-Mask-Detection-master\dataset"
categories = ["with_mask", "without_mask"]

print("Loading images...")

data = []
labels = []

for category in categories:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

#using one-hot encoding for the labels
labelbinarizer = LabelBinarizer()
labels = labelbinarizer.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype ="float32")
labels = np.array(labels)

#splitting into training and testing data
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

#loading MobileNEtV2 model, leaving the head FC layer sets
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

#constructing head of the model placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#joining head model and base model
model = Model(inputs=baseModel.input, outputs=headModel)

#looping over all layers so that they are not updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

#compiling the model
print("Compiling model...")
opt = Adam(learning_rate=learningrate, decay=learningrate/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#train the head of the network
print("Training the model head")
Head = model.fit(aug.flow(trainX, trainY, batch_size=batchsize), steps_per_epoch=len(trainX)//batchsize, validation_data=(testX, testY), validation_steps=len(testX)//batchsize, epochs = epochs)

#making predictions
print("Evaluating network...")
predictions = model.predict(testX, batch_size=batchsize)

#finding largest predicted probability
predictions = np.argmax(predictions, axis=1)

#showing classification report
print(classification_report(testY.argmax(axis=1), predictions, target_names=labelbinarizer.classes_))

#Saving the model to disk
print("Saving model to disk...")
model.save("mask_detector.model", save_format="h5")

#plot the training loss and accuracy
n = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,n), Head.history["loss"], label="Training Loss")
plt.plot(np.arange(0,n), Head.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0,n), Head.history["accuracy"], label="Training Accuracy")
plt.plot(np.arange(0,n), Head.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")















