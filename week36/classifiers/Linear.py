import numpy as np

class LinearClassifier:
    """
    Base class for all self made linear classifiers
    
    Parameters:
        Xtr1: array_like
            Trainingdata for class 1
        Xtr2: array_like
            Trainingdata for class 2
    """

    def __init__(self, Xtr1, Xtr2):
        self.Xtr1, self.Xtr2 = Xtr1, Xtr2
        self.Ytr = np.concatenate((np.ones(len(Xtr1)), np.ones(len(Xtr2)) * -1))
        self.Xtr = np.concatenate((Xtr1, Xtr2))
        self.Xtr = np.append(self.Xtr, np.ones((len(self.Xtr), 1)), axis=1)
        self.Xtr, self.Ytr = self.shuffle(self.Xtr, self.Ytr)

        self.trained = False
        self.tested = False

        self.w = np.random.uniform(size=(3, 1))


    def is_trained(self):
        if not self.trained:
            print("Model not been trained! Use class.train(rho)")
            exit()
    
    def is_tested(self):
        if not self.tested:
            print("Model has not been tested! Use class.test()")
            exit()

    @staticmethod
    def shuffle(data, labels):
        ind = np.arange(len(data))
        np.random.shuffle(ind)
        data = data[ind]
        labels = labels[ind]
        return data, labels