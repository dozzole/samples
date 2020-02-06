import re
import numpy as np
import itertools as it
from random import random
from sympy import Symbol, poly

class SymmetryGroup(object):
    """
    Class that contains symmetry groups
    :param descr: Name of the symmetry group
    :type descr: str
    :param listofgenerators: Python list containing the generators of the group
    :type listofgenerators: list
    """
    def __init__(self, descr, listofgenerators):
        self.__generators = listofgenerators
        self.__name = descr
        self.numgens = len(listofgenerators)
        self.__buildGroup()
        self.numelem = len(self.__transforms)

    @classmethod
    def isDummy(cls, descr):
        """
        Classmethod for isotropic symmetry group. Creates a dummy SymmetryGroup object.
        :param descr: Name of the symmetry group
        :return: None
        """
        return cls(descr, [np.identity(3)])

    def __buildGroup(self):
        """
        Builds symmetry group from listofgenerators
        :return: list of symmetry transformations
        """
        self.__transforms = self.__generators
        startLength = self.numgens
        endLength = 10000
        while startLength != endLength:
            startLength = len(self.__transforms)
            for t1 in self.__transforms:
                for t2 in self.__transforms:
                    cand = t1@t2
                    add = True
                    for arr in self.__transforms:
                        if np.isclose(arr, cand).all():
                            add = False
                            break
                    if add:
                        self.__transforms.append(cand)
            endLength = len(self.__transforms)

    def getGenerators(self):
        return self.__generators

    def getName(self):
        return self.__name

    def getElements(self):
        return self.__transforms

