import re
import numpy as np
import itertools as it
from random import random
from sympy import Symbol, poly

class FitData(dict):
    def __init__(self, fname, path="output/"):
        self.__data = {}
        self.__name = fname
        self.__file = path + fname.replace(' ', '_')


    def __setitem__(self, key, value):
        if key is not "FitMatrix":
            key = str(key)
            value = str(value)

        try:
            file = open(self.__file, "r")
            file.close()
        except FileNotFoundError:
            print("Create file: " + self.__file)
            file = open(self.__file, "w")
            file.close()

        if key in self.keys():
            keyexists = True
        else:
            keyexists = False

        if not keyexists:
            with open(self.__file, "a") as file:
                file.write(key + "=" + value + "\n")
        else:
            linenumber = 9999
            with open(self.__file, "r") as file:
                data = file.readlines()
                for num, line in enumerate(data, 1):
                    if line.startswith(key):
                        linenumber = num
                        break

            data[linenumber-1] = key + "=" + value + "\n"

            with open(self.__file, "w") as file:
                file.writelines(data)

    def __getitem__(self, item):
        with open(self.__file, "r") as file:
            for line in file:
                if line.startswith(item):
                    return line.split('=')[1].strip("\n")
            raise KeyError(str(item) + " is not a key in the dictionary")

    def __str__(self):
        return str(self.getDict())

    def keys(self):
        with open(self.__file, "r") as file:
            lines = file.readlines()
            keys = map(lambda x: x.split('=')[0], lines)
        return list(keys)

    def getDict(self):
        for key in self.keys():
            self.__data.update({key : self.__getitem__(key)})
        return self.__data

    def items(self):
        return self.getDict().items()

    def getCoeffs(self, pat = r'c\d{4}'):
        coeffs = []
        regex = re.compile(pat)
        for key in self.keys():
            if regex.search(key):
                coeffs.append(key)
        return coeffs

    def getFitResult(self, clean=True, pat=r'\w\d{1,4}'):
        if clean:
            matrixkey = "FitMatrixClean"
        else:
            matrixkey = "FitMatrix"
        mat = list(self.__getitem__(matrixkey))
        mat = ''.join(mat)

        d = self.getDict()

        # for key, val in d.items():
        #     print(str(key) + " : " + str(val))

        def fun(xx):
            # print(xx.group(0))
            return d[xx.group(0)]

        mat = re.sub(pat, fun, mat)

        mat = mat.split('],[')

        for i, line in enumerate(mat):
            mat[i] = line.replace('[[','').replace(']]','')
            mat[i] = mat[i].split(',')


        for i, row in enumerate(mat):
            for j, col in enumerate(row):
                mat[i][j] = eval(mat[i][j])
        mat = np.asarray(mat)
        return mat

    def getBaseMatrix(self):
        coeffs = list(self.keys())
        notcoeffs = ["FitMatrix", "FitMatrixClean"]
        coeffs = [elem for elem in coeffs if elem not in notcoeffs]
        coeffgroups = [(coeffs[0][0], [coeffs[0]])]
        for elem in coeffs:
            foolist = [elem[0] == group[0] for group in coeffgroups]
            if not any(foolist):
                coeffgroups.append((elem[0], [elem]))
            else:
                for group in coeffgroups:
                    if elem[0] == group[0] and elem not in group[1]:
                        group[1].append(elem)

        reslist = list()

        for group in coeffgroups:
            p1 = float(self.__getitem__(str(group[1][0])))
            p2 = float(self.__getitem__(str(group[1][1])))
            p3 = float(self.__getitem__(str(group[1][2])))
            p4 = float(self.__getitem__(str(group[1][3])))
            p5 = float(self.__getitem__(str(group[1][4])))
            mat = np.array([
                            [p1, p2, p3, 0, 0, 0],
                            [p2, p1, p3, 0, 0, 0],
                            [p3, p3, p4, 0, 0, 0],
                            [0, 0, 0, 2 * p5, 0, 0],
                            [0, 0, 0, 0, 2 * p5, 0],
                            [0, 0, 0, 0, 0, p1 - p2]
                        ])

            reslist.append((group[1], mat))
        return reslist


    def getName(self):
        return self.__name
