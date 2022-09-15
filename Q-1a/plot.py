import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt

def cm_plot(y_actual, y_predict):
    confusion_matrix = sm.confusion_matrix(y_actual, y_predict)
    ticks = ['setosa', 'versicolor', 'virginica']

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
    ax.set_xticks(np.arange(len(ticks)), labels=ticks)
    ax.set_yticks(np.arange(len(ticks)), labels=ticks)

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(x=j, y=i, s=confusion_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.close()
    return fig

def loss_plot(history):
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, loss, 'bo')
    plt.plot(epochs, loss, 'bo', label="Training loss")
    plt.plot(epochs, val_loss, 'b', label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.close()
    return fig

def accuracy_plot(history):
    plt.clf()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, acc, 'bo')
    plt.plot(epochs, acc, 'bo', label="Training acc")
    plt.plot(epochs, val_acc, 'b', label="Validation acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.close()
    return fig