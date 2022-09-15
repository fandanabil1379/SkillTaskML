from keras.models import Sequential 
from keras.utils import np_utils
from keras.layers import Dense, Dropout 

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

featureName = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
targetName  = ['species']

def baseModel():
    model = Sequential()
    model.add(Dense(1000, input_dim=4, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def prepData(data, testsize=0.2, **kwargs):
    X = data[featureName].values
    y = data[targetName].values
    Xnorm = normalize(X, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(Xnorm, y, test_size=testsize, **kwargs)
    y_train = np_utils.to_categorical(y_train, num_classes=3)
    y_test = np_utils.to_categorical(y_test, num_classes=3)
    return {
        "Xtrain": X_train.tolist(),
        "ytrain": y_train.tolist(),
        "Xtest": X_test.tolist(),
        "ytest": y_test.tolist(),
    }