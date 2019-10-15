#to encode image pixels 1 to 0
from sklearn.preprocessing import LabelEncoder
#k fold split test and validation set to classify data is close result 
from sklearn.model_selection import train_test_split
#to shuffle arrays or sparse matrices in a consistent way
from sklearn.utils import shuffle
#to draw loss and accuracy 
import matplotlib.pyplot as plt

#to change learning rate.gradient descent using.
from keras import layers, optimizers, models
#to use l2 penalty regularization
from keras.constraints import maxnorm
#to classifying more than one person so using categorical
from keras.utils import to_categorical
#to prevent overfitting data.Close to real image
from keras.preprocessing.image import ImageDataGenerator
#very usefull l2 penalty adaptive learning rate.
from keras.optimizers import RMSprop,Adam
#collection of algorithms for image processing. 
from skimage import io
#to use array
import numpy as np
#to draw line resize ,crop image
import cv2
#to acces os system for example mkdir to create folder.
import os
#to play voice
from pygame import mixer
#to sleep program 
import time

#50x50 pixel images setting width and height
width_height = 50
#my img folder in directory names
dirs = ['abel','abraham','adler','adriel','ali','aron','benson','bryan','cadman','casey','dante','erdogan','mert','erdogan2']
#encode directory name to binary
encoder = LabelEncoder()
#apply encoding directory name
y = encoder.fit_transform(dirs)

#to train our model
def train():
    global y
    x = None

    if not os.path.exists('dataset.npy'):

        index = 1
        imgs = []
        for dir in dirs:
            #acces images 
            images = os.listdir(os.path.join("img", dir))
            for image in images:
                #check file extension.
                if image.endswith(".png"):
                    #convert image to pixel with prepare_image (50, 50, 1)
                    tmp_img = prepare_image(os.path.join("img", dir, image))
                    #print(tmp_img.shape)
                    #to desired image size and change shape to give cnn filters and ann(artificall neural network)
                    tmp_img = resize(width_height, width_height, tmp_img)
                    #reshape to desired pixel
                    tmp_img = tmp_img.reshape((tmp_img.shape[0] * tmp_img.shape[1],))
                    b = np.zeros((tmp_img.shape[0] + 1,))
                    b[0] = y[index - 1]
                    b[1:] = tmp_img
                    imgs.append(b)
            index += 1
        
        x = np.array(imgs)       
        #shuffle pixels learn different images to prevent overfitting
        x = shuffle(x)
        #to save learning pixel which means that every pixel coded in machine
        np.save("dataset", x)
    else:
        x = np.load("dataset.npy")
    #pixels for cnn convert shape.    
    y = x[:, [0]]
    y = y.reshape(y.shape[0], )
    x = x[:, 1:]
    #my classifed size.which means that if ı 5 person ,i clasified 5 perosn
    total = len(dirs)
    #our pixels
    x = x.astype('float64')
    x = x.reshape(x.shape[0], width_height, width_height, 1)
    #people corresponding to pixels
    y = to_categorical(y, total)

    model = models.Sequential()
    #this is first neuron it helps to 32 filter 5x5 dimension matris ,stride 1 default , maxnorm(m) will, if the L2-Norm of your weights exceeds m, scale to change variance limited.
    #input shape our data shape 50x50 and 1 means is we use gray scale to machine easly train data
    model.add(layers.Conv2D(32, (5, 5), padding='same', kernel_constraint=maxnorm(3), input_shape=(width_height, width_height, 1)))
    #activation function is relu because easy to derivate function and if you pass greater than 0 you pass the activation and second layer.
    model.add(layers.Activation('relu'))
    #pooling layers reduce the dimensions of the data by combining the outputs of neuron clusters at one layer into a single neuron in the next layer. ...
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    #same layer adding.More layers means more detail.
    model.add(layers.Conv2D(64, (5, 5), padding='same', kernel_constraint=maxnorm(3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #same layer adding.More layers means more detail.
    model.add(layers.Conv2D(32, (5, 5), padding='same', kernel_constraint=maxnorm(3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    #all image pixel give to neural network to classifier all images.
    model.add(layers.Flatten())
    #my basic ANN (artifical neural network) macnorm is a l2 penalty ridge regresion
    model.add(layers.Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    #In this layer, some neurons in our network are randomly disabled.
    model.add(layers.Dropout(0.5))
    
    #our data is categorical so we use softmax it similar with sigmoid but sigmoid only use binary classification.We need to use categorical function = softmax.
    #we can only output up to the number of pictures.so total is number of img directory size
    model.add(layers.Dense(total, activation='softmax'))
    #epoch means number of iteration of neurons.if we increase epoch size ,accuracy will increase
    epochs = 50
    #every time take 32 images feed network with backpropagation
    batch_size = 32
    #optimizers=Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999), metrics=['accuracy'])
    #train %70 test %30 and shuffle train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.3)
    #model.fit(x_train, y_train, validation_split=0.3, epochs=epochs, batch_size=32)
    #to prevent data overfitting i generate sam photo with rotation,shortly same data extended
    datagen=ImageDataGenerator(featurewise_center=False, #set input mean to 0
                           samplewise_center=False,  #set each sample mean to 0
                           featurewise_std_normalization=False, #divide input datas to std
                           samplewise_std_normalization=False,  #divide each datas to own std
                           zca_whitening=False,  #dimension reduction
                           rotation_range=40,    #rotate 90 degree
                           zoom_range=0.5,        #zoom in-out 5%
                           width_shift_range=0.5, #shift 5%
                           height_shift_range=0.5,
                           horizontal_flip=True,  #randomly flip images
                           vertical_flip=False,
                           shear_range=0.01,
                           data_format='channels_last',
                           #brightness_range=[0.2,0.9] if we add brightness our accuracy decrase wo i cancelled this option
                           ) 
    #all train data extended programmaticly to escape overfitting.
    datagen.fit(x_train)
    history=model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),epochs=epochs,
                            validation_data=(x_test,y_test),steps_per_epoch=x_train.shape[0]//batch_size)
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    #images augmentation to avoid overfitting all images 
    if not os.path.exists('preview_dataGenerator'):
        os.mkdir('preview_dataGenerator')        
        i = 0
        for batch in datagen.flow(x, batch_size=1,save_to_dir='preview_dataGenerator', save_prefix='face', save_format='png'):   
            i += 1
            #its test data to show first 20 picture.
            if i > 20:
                break  # otherwise the generator would loop indefinitely 
    #showing model accuracy     
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test Accuracy : " + str(accuracy))
    print("Test Loss : " + str(loss))
    
    #keras library helps to create updated w all information automic create.
    model.save_weights('face.h5')
    #all faces converted readable file to detect data.
    with open('face.json', 'w') as f:
        f.write(model.to_json())

#to scale image to prepare train
def resize(max_height: int, max_width: int, frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]

    if max_height < height or max_width < width:
        if width < height:
            scaling_factor = max_width / float(width)
        else:
            scaling_factor = max_height / float(height)

        frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

#crop  face(images) in center
def crop_center(frame: np.ndarray) -> np.ndarray:
    short_edge = min(frame.shape[:2])
    yy = int((frame.shape[0] - short_edge) / 2)
    xx = int((frame.shape[1] - short_edge) / 2)
    crop_img = frame[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

    start = int((frame.shape[1] - frame.shape[0]) / 2)
    end = int(frame.shape[1] - (frame.shape[1] - frame.shape[0]) / 2)
    return frame[:, start:end]

#detect face with model 
def detect_my_face():
    #face recognition with voice
    voice_active = input("Do you want to voice activate ?  [Y]/n ")
    voice_active.lower()
    if(voice_active == 'y'):
        print("Voice Activated.")
    else:
        print("Voice Deactivated")
    # capture frames from a camera
    cap = cv2.VideoCapture(0)
    #creation dense in this part we create dense model to detect images(faces)
    model = models.Sequential()
    #our faces
    with open('face.json', 'r') as f:
        model = models.model_from_json(f.read())
    #our w which means that weights model 
    model.load_weights('face.h5')

    while (1):
        #read camera frames
        ret, real_frame = cap.read()
        #convert rgb to grayscale
        frame = cv2.cvtColor(real_frame, cv2.COLOR_BGR2GRAY)
        #resize and crop image to recognize spesification
        frame = resize(width_height, width_height, frame)
        frame = crop_center(frame)
        #convert binary to normalize data
        predict = frame / 255
        predict = predict.astype('float64')
        #convert rgb to gray 
        predict = predict.reshape(width_height, width_height, 1)
        #predict the frame(image captured) according to model
        prediction = model.predict(np.array([predict])).tolist()
        #if prediction result greater than our trashold result then prediction is correct
        prediction_result = np.max(prediction)
            #to get accurate person(trashold level=limit value)
        if prediction_result > 0.5:
            #directory name which means that predicted names
            text = dirs[np.argmax(prediction)]
            #convert prediction result 
            prediction_result = "%.2f" % prediction_result
            prediction_result = int(float(prediction_result)*100)
            #to draw line we define start line and end line
            start = int((real_frame.shape[1] - real_frame.shape[0]) / 1.5)
            end = int(real_frame.shape[1] - (real_frame.shape[1] - real_frame.shape[0]) / 1.5)
            #position of the prediction text
            cv2.putText(real_frame, text, (0, 160), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 255))
            cv2.putText(real_frame, "%"+str(prediction_result), (0, 55), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 255))
            #if erdogan program says i recognize you if not access is denied
            if text == "erdogan":
                cv2.putText(real_frame, "i recognize you", (0, 100), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0))
                print("Ok.. I recognize you.. Welcome")
                print("To close camera push ESC ")
            else:
                cv2.putText(real_frame, "Access is denied", (0, 100), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0))
                print("Who you are? Access is denied")
                print("To close camera push ESC ")
            #voice activation 
            if voice_active == "y":
                if text == "erdogan":
                    mixer.init()
                    mixer.music.load("know.mp3")
                    mixer.music.play()
                    time.sleep(5)
                                   
                else:
                    mixer.init()
                    mixer.music.load("denied.mp3")
                    mixer.music.play()
                    time.sleep(5)
                    
                
        else:
            #if predicted model cant find anyone, it says no one
            cv2.putText(real_frame, "NO ONE", (0, 125), cv2.FONT_HERSHEY_DUPLEX, 1.6, (50,255,0))
        #line weights with open cv    
        lineThickness = 5
        cv2.line(real_frame, (start, 0), (start, real_frame.shape[0]), (50,255,0), lineThickness)
        cv2.line(real_frame, (end, 0), (end, real_frame.shape[0]), (50,255,0), lineThickness)

        real_frame = resize(350, 350, real_frame)
        cv2.imshow("Capture Camera Recognition", real_frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

#to create image data my own datas
def save_images(path):
    print('Loading ' + path + "image")
    cap = cv2.VideoCapture(0)
    i = 0
    while True:
        print(str(i))
        i = i + 1

        ret, frame = cap.read()
        frame = resize(width_height, width_height, frame)
        frame = crop_center(frame)

        cv2.imshow("Shrinked image", frame)
        cv2.imwrite('img/' + path + '/' + str(i) + '.png', frame)
        #exit with esc
        cv2.waitKey(5) & 0xFF
        #when take 1000 images break
        if i >= 1001:
            break

    cap.release()
    cv2.destroyAllWindows()

#to convert image to pixel
def prepare_image(image_path: str) -> np.ndarray:
    tmp_img = io.imread(image_path)
    #I chose grayscale to avoid fatigue.
    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
    tmp_img = resize(width_height, width_height, tmp_img)
    tmp_img = crop_center(tmp_img)
    #all image normalize with  encode 0 to 1 
    tmp_img = tmp_img / 255
    tmp_img = tmp_img.astype('float64')
    
    tmp_img = tmp_img.reshape(width_height, width_height, 1)
    return tmp_img

#call save image to save image in directories
def collect_camera_images():
    for people in dirs:
        save_images(people)

#to create data my own images
#collect_camera_images()
#train with collected images taken with camera automaticly
#train()
#detect face with dataset using face h5 and json file
detect_my_face()
