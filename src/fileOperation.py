from Logger import MyLogger
from copy import deepcopy
import numpy as np

class FileManager:
    def __init__(self):
        self.logger = MyLogger().getLogger()

    def readData(self, fileName):
        Dataset = open('../DataSet_DataBase/' + fileName, 'r')
        data = Dataset.readlines()
        Array = []
        T = ","
        # ecoli Data set separated by double empty space
        if fileName == "ecoli":
            T = "  "
        for line in data:
            # remove \n char from the end of the text ,
            # then split the row depends on comma
            record = line[:-1].split(T)
            if '?' not in record: Array.append(record)
        self.logger.info(
            "Reading Data set:{} Feature Numbers:{} Number of rows:{}".format(fileName, len(Array[0]), len(Array)))
        return Array

    def moveTargetToLastColumn(self, data):
        modifyData = deepcopy(data)
        row = len(modifyData)
        column = len(modifyData[0])
        for i in range(row):
            temp = modifyData[i][0]
            modifyData[i][0] = modifyData[i][column - 1]
            modifyData[i][column - 1] = temp
        self.logger.info("Move first column to last column")
        return modifyData

    def deleteColumn(self, data, index):
        modifyData = deepcopy(data)
        for row in modifyData:
            del row[index]
        self.logger.info("Delete index {} from data set".format(index))
        return modifyData

    def readColumn(data, index):
        s = []
        for row in data:
            s.append(row[index])
        return s

    def convertContToDisc(self, data, TargetColumnsIndex):
        copyData = deepcopy(data)
        for column in TargetColumnsIndex:
            # Read Column
            columnData = self.readColumn(data, column)
            print("Column :", column, "len: ", len(set(columnData)))
            if len(set(columnData)) < 10:
                continue
            columnData = list(map(float, columnData))
            rangeP = max(columnData) - min(columnData)
            Interval = rangeP / 5.0
            Intervals = list(np.arange(min(columnData), max(columnData) + Interval, Interval))
            print("Column: ", column, " Int:", Intervals)
            EditedCol = []
            for val in columnData:
                for t in range(len(Intervals)):
                    if t == len(Intervals) - 1:
                        EditedCol.append(t)
                        continue
                    if Intervals[t] <= val < Intervals[t + 1]:
                        EditedCol.append(t)
                        continue

            # Save Column
            for row in copyData:
                row[column] = str(EditedCol[column])
        print(copyData)
        return copyData

    def convertDiscreateToNumbers(self, data):
        copyData = deepcopy(data)
        NumberOfColumn = len(data[0])
        for i in range(NumberOfColumn):
            cat_MAP = list(set(self.readColumn(data, i)))
            print("Index ", i, "CAT", cat_MAP)
            ColumnData = self.readColumn(data, i)
            # Edit Data
            for k in range(len(ColumnData)):
                ColumnData[k] = cat_MAP.index(ColumnData[k])
            # Save column
            count = 0
            for row in copyData:
                row[i] = ColumnData[count]
                count += 1

            # print("Before:", self.readColumn(data, i))
            # print("After: ", self.readColumn(copyData, i))

        return copyData

    def ScaleArray(self, data, TargetColumns):
        copyData = deepcopy(data)
        for i in TargetColumns:
            cat_MAP = list(set(self.readColumn(data, i)))
            if len(cat_MAP) == 1:
                continue
            minVal = min(cat_MAP)
            maxVal = max(cat_MAP)
            # print("Max ", maxVal, " Min ", minVal)
            ColumnData = self.readColumn(data, i)
            # Edit Data
            for k in range(len(ColumnData)):
                ColumnData[k] = ((ColumnData[k] - minVal) * ((1 - 0) / (maxVal - minVal))) + 0
            # Save column
            count = 0
            for row in copyData:
                row[i] = ColumnData[count]
                count += 1
        return copyData
