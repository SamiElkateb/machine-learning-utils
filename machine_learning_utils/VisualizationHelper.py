from dataclasses import dataclass
from typing import List

import numpy as np
from keras.api._v2.keras import Model
from keras.api._v2.keras.callbacks import History
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


@dataclass
class ModelHistory:
    def __init__(self, model: Model, history: History, name: str) -> None:
        self.model = model
        self.history = history
        self.name = name


class VisualizationHelper:
    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        self.update_data(x_train, y_train, x_test, y_test)
        self.results: List[ModelHistory] = []

    def update_data(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def add_history(self, model: Model, history: History, name: str) -> None:
        self.results.append(ModelHistory(model, history, name))

    def plot_results(self, number=1, title="Your Plot Title") -> None:
        t_number = number
        if t_number > len(self.results):
            t_number = len(self.results)
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(title, fontsize=16, y=1.05)

        for i in range(t_number):
            history = self.results[-i].history
            name = self.results[-i].name
            subplot1 = fig.add_subplot(1, 3, 1)
            subplot1.plot(history.history["accuracy"], label=f"{name}_{i}: accuracy")
            subplot1.plot(
                history.history["val_accuracy"], label=f"{name}_{i}: val_accuracy"
            )
            subplot1.set_title("Accuracy")
            subplot1.legend()

        # Loss plot
        for i in range(t_number):
            history = self.results[-i].history
            name = self.results[-i].name
            subplot2 = fig.add_subplot(1, 3, 2)
            subplot2.plot(history.history["loss"], label=f"{name}_{i}: loss")
            subplot2.plot(history.history["val_loss"], label=f"{name}_{i}: val_loss")
            subplot2.set_title("Loss")
            subplot2.legend()

        for i in range(t_number):
            if i > 1:
                break
            model = self.results[-i].model
            y_pred = model.predict(self.x_test)
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(self.y_test, axis=1)
            calculated_f1_score = f1_score(y_test, y_pred, average="macro")
            fig.text(
                0.5,
                0.95,
                f"F1-Score: {calculated_f1_score}",
                ha="center",
                va="center",
                fontsize=12,
            )
            # Confusion matrix
            calculated_confusion_matrix = confusion_matrix(y_test, y_pred)
            subplot3 = fig.add_subplot(1, 3, 3)
            caxes = subplot3.matshow(calculated_confusion_matrix)
            for (i, j), z in np.ndenumerate(calculated_confusion_matrix):
                subplot3.text(j, i, "{:0.1f}".format(z), ha="center", va="center")
            fig.colorbar(caxes, ax=subplot3)

        plt.tight_layout()
        plt.show()
