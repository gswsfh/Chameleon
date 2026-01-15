
from sklearn.model_selection import train_test_split


def splitDataset(data,label,test_size=0.3):
    x_train, x_test, y_train, y_test = train_test_split(data, label,test_size=test_size)
    return x_train, x_test, y_train, y_test