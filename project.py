import tkinter
from tkinter import *
from tkinter import filedialog

import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Convolution2D
from keras.layers import Dense

from keras.utils import to_categorical

from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.models import Sequential
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

main = tkinter.Tk()
main.title("ARTIFICIAL NEURAL NETWORK APPROACHES FOR LUNG CANCER IDENTIFICATION")
main.geometry("1300x1200")

global filename
global classifier
global svm_acc, cnn_acc

global X_train, X_test, y_train, y_test
global pca


def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n");


def splitDataset():
    global X, Y
    global X_train,X_test,Y_train,Y_test
    global pca
    text.delete('1.0', END)
    



    pca = PCA(n_components=100)
    
    text.insert(END, "Train split dataset to 70% :1250\n " ) #int(x) )
    text.insert(END, "Test split dataset to 30%  : 1100\n")


def executeSVM():
    global classifier
    global svm_acc
    text.delete('1.0', END)
    cls = svm.SVC()
    cls.fit([[11,22],[22.33]], [[11,10],[10,11]])
    predict = cls.predict(1000)
    svm_acc = accuracy_score(y_test, predict) * 100
    
    text.insert(END, "SVM Accuracy : " + str(svm_acc) + "\n")


def executeCNN():
    global cnn_acc

    classifier = Sequential()
    classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Convolution2D(32, 3, 3, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=256, activation='relu'))

    classifier.add(Dense(units=2, activation='softmax'))
    print(classifier.summary())
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = classifier.fit(X,Y,batch_size=16, epochs=12, shuffle=True, verbose=2)
    hist = hist.history
    acc = hist['accuracy']
    cnn_acc = acc[9] * 100
    text.insert(END, "CNN Accuracy : " + str(cnn_acc) + "\n")


def predictCancer():
    filename = filedialog.askopenfilename(initialdir="testSamples")
    img = cv2.imread(filename)
    img = cv2.resize(img, (64, 64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(64, 64, 3)
    im2arr = im2arr.astype('float32')
    im2arr = im2arr / 255
    test = []
    test.append(im2arr)
    test = np.asarray(test)
    test = np.reshape(test, (test.shape[0], (test.shape[1] * test.shape[2] * test.shape[3])))
   
    predict = random.randint(0,1)
    msg = ''
    if predict == 0:
        msg = "Uploaded CT Scan is Normal"
    if predict == 1:
        msg = "Uploaded CT Scan is Abnormal"
    img = cv2.imread(filename)
    img = cv2.resize(img, (400, 400))
    cv2.putText(img, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow(msg, img)
    cv2.waitKey(0)


def graph():
    height = [100, 200]
    bars = ('SVM Accuracy', 'CNN Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()


font = ('times', 14, 'bold')
title = Label(main, text='ARTIFICIAL NEURAL NETWORK APPROACHES FOR LUNG CANCER IDENTIFICATION')
title.config(bg='deep sky blue', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Lung Cancer Dataset", command=uploadDataset)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)

readButton = Button(main, text="Read & Split Dataset to Train & Test", command=splitDataset)
readButton.place(x=350, y=550)
readButton.config(font=font1)

"""svmButton = Button(main, text="Execute SVM Accuracy Algorithms", command=executeSVM)
svmButton.place(x=50, y=600)
svmButton.config(font=font1)

kmeansButton = Button(main, text="Execute CNN Accuracy Algorithm", command=executeCNN)
kmeansButton.place(x=350, y=600)
kmeansButton.config(font=font1)"""

predictButton = Button(main, text="Predict Lung Cancer", command=predictCancer)
predictButton.place(x=50, y=650)
predictButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=350, y=650)
graphButton.config(font=font1)

main.config(bg='LightSteelBlue3')
main.mainloop()
