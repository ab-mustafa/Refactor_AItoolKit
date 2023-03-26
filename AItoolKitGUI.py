from __future__ import annotations
import tkinter as tk
from abc import ABC, abstractmethod
import re
# Client Code ...
# This code need to be inside AIFactory

from src.AI_Factory import AIFactory
from src.CrossValidation import CrossValidation
from src.DataSet import DataSet
from src.ProductsCreator import ID3BoostCreator
from src.ProductsCreator import NaiveBayesCreator
from src.Logger import MyLogger
import sys
import statistics
from src.fileOperation import FileManager, ExcelDataSet, TextDataSet


class Mediator(ABC):
    @abstractmethod
    def notify(self, sender: object, event: str) -> None:
        pass


class BaseComponent:
    def __init__(self, mediator: Mediator):
        self._mediator = mediator


class DropdownList(BaseComponent):
    def __init__(self, mediator: Mediator, name: str, options: list, parent: tk.Frame, rows: int, columns: int):
        super().__init__(mediator)
        self._name = name
        self._options = options
        self.rows = rows
        self.columns = columns
        self._selected = tk.StringVar()
        self._selected.set(options[0])


        self._menu = tk.OptionMenu(parent, self._selected, *options, command=self._on_select)
        self._menu.grid(row=rows, column=columns, padx=10, pady=10)

    def _on_select(self, selected_option):
        self._mediator.notify(self, selected_option)

    @property
    def selected(self):
        return self._selected.get()


class Label(BaseComponent):
    def __init__(self, mediator: Mediator, name: str, parent: tk.Frame, rows: int, columns: int):
        super().__init__(mediator)
        self._name = name
        self.rows = rows
        self.columns = columns
        self._label = tk.Label(parent, text=f"{name}:")
        self._label.grid(row=rows, column=columns, padx=10, pady=10)

    def recieveText(self, newText: str):
        self._label.config(text=newText)


class NumberInput(BaseComponent):
    def __init__(self, mediator: Mediator, name: str, parent: tk.Frame, rows: int, columns: int):
        super().__init__(mediator)
        self._name = name
        self.rows = rows
        self.columns = columns
        self._entry = tk.Entry(parent, width=10)
        self._entry.grid(row=rows, column=columns, padx=10, pady=10)

    def get_input(self):
        return self._entry.get()


class TrainButton(BaseComponent):
    def __init__(self, mediator: Mediator, parent: tk.Frame):
        super().__init__(mediator)

        self._button = tk.Button(parent, text="Train", command=self._on_train_click)
        self._button.grid(row=5, column=1, padx=10, pady=10)

    def _on_train_click(self):
        n_folds = int(self._mediator._n_folds_input.get_input())
        n_iters = int(self._mediator._n_iters_input.get_input())
        self._mediator.notify(self, "train")
        print(
            f"Training {self._mediator._algorithm} on {self._mediator._dataset} with {n_folds} folds and {n_iters} iterations.")
        print("Code running")

        if re.search(r"xlsx$", self._mediator._dataset):
            Data = ExcelDataSet()
        else:
            Data = TextDataSet()

        Data.startProcessing(self._mediator._dataset)
        # Data = FileManager(self._mediator._dataset)

        dataSet = DataSet(Data.getDataSet())
        CrossValid = CrossValidation(n_iters, n_folds, dataSet, self._mediator._algorithm)
        CrossValid.dataShufflingDT()
        result = CrossValid.triggerCrossValidation()
        self.average_Accuracy = sum(result) / len(result)
        self.std_dev = statistics.stdev(result)
        print("Standard Deviation: {}".format(self.std_dev))

        print("Average Accuracy: {}".format(self.average_Accuracy))
        self._mediator.notify(self, "TrainEvent")



class AIToolkit(Mediator):
    def __init__(self, parent: tk.Frame):
        self._dataset = None
        self._algorithm = None
        self._n_folds = None
        self._n_iters = None

        self.datasetLabel = Label(self, "DataSet", parent, 0, 0)
        self._dataset_dropdown = DropdownList(self, "Dataset",
                                              ["cancer.data", "car.data", "letter.data", "mushroom.data",
                                               "tennis.csv", "play_tennis.xlsx"], parent, 0, 1)

        self.algorithmLabel = Label(self, "Algorithm", parent, 1, 0)
        self._algorithm_dropdown = DropdownList(self, "Algorithm", ["adaboost", "naivebayes"], parent, 1, 1)

        self.foldLabel = Label(self, "Number of Folds", parent, 2, 0)
        self._n_folds_input = NumberInput(self, "Number of Folds", parent, 2, 1)

        self.itersLabel = Label(self, "Number of Iterations", parent, 3, 0)
        self._n_iters_input = NumberInput(self, "Number of Iterations", parent, 3, 1)

        self._train_button = TrainButton(self, parent)

        self._status_label = Label(self, "Please select a dataset, an algorithm, number of folds, and number of iterations.",parent,6,1)


    def notify(self, sender: object, event: str) -> None:
        if isinstance(sender, DropdownList):
            if sender._name == "Dataset":
                self._dataset = sender.selected
            elif sender._name == "Algorithm":
                self._algorithm = sender.selected


        elif event == "TrainEvent":
            Text = "Average Accuracy {}    Standard Deviation {}".format(sender.average_Accuracy, sender.std_dev)
            self._status_label.recieveText(Text)
        elif isinstance(sender, NumberInput):
            if sender._name == "Number of Folds":
                self._n_folds = int(sender.get_input())
            elif sender._name == "Number of Iterations":
                self._n_iters = int(sender.get_input())

        elif isinstance(sender, TrainButton) and event == "train":
            if (self._dataset is None or self._algorithm is None or
                    self._n_folds is None or self._n_iters is None):
                self._status_label.recieveText(f"Training {self._algorithm} on {self._dataset} with {str(self._n_folds)} folds and {str(self._n_iters)}")

            else:
                self._status_label.recieveText(f"Training {self._algorithm} on {self._dataset} with {self._n_folds} folds and {self._n_iters} iterations...")





if __name__ == "__main__":
    root = tk.Tk()
    # root.title("AI Toolkit")
    root.geometry("800x600")

    app_frame = tk.Frame(root)
    # app_frame.pack()

    ai_toolkit = AIToolkit(root)

    root.mainloop()
