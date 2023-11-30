import pandas as pd
import matplotlib.pyplot as plt
import time


class ProgressMeter:
    def __init__(self, filename, name="Loss"):
        self.epochs = []
        self.train_values = []
        self.val_values = []
        self.timestamps = []
        self.name = name
        self.filename = filename

    def pushback(self, epoch, train_value, val_value, display=False):
        self.train_values.append(train_value)
        self.val_values.append(val_value)
        self.epochs.append(epoch)
        self.timestamps.append(time.time())
        if display:
            print(("Epoch {},\tTraining " + self.name + ": {},\tValidation " + self.name + ":{}").format(
                epoch, self.train_values[-1], self.val_values[-1]))

    def removeNLastEpochs(self, N):
        print("Removing the {} last epochs due to ealy stopping".format(N))
        self.train_values = self.train_values[:-N]
        self.val_values = self.val_values[:-N]
        self.epochs = self.epochs[:-N]
        self.timestamps = self.timestamps[:-N]

    def __update__df(self):
        self.df = pd.DataFrame({"Epochs": self.epochs, "Training": self.train_values,
                                "Validation": self.val_values, "Timestamps": self.timestamps})

    def plot(self, ax=None, title="Training Curve"):
        self.__update__df()
        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.plot(self.df["Epochs"].values, self.df["Training"].values, label="Training " + self.name)
        ax.plot(self.df["Epochs"].values, self.df["Validation"].values, label="Validation " + self.name)
        ax.set_xlabel("Epoche")
        ax.set_ylabel(self.name)
        ax.legend()
        ax.set_title(title)

    def save(self):
        self.__update__df()
        self.df.to_csv(self.filename)
