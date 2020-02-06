import re
import numpy as np
import itertools as it
from random import random
from sympy import Symbol, poly

class Stiffness(object):
    """
    Class that builds stiffness tensors
    :param m: Voigt-representation of a stiffness
    :type m: numpy array (6x6)
    :param descr: Descriptor of the stiffness
    :type descr: str
    """

    def __init__(self, m, descr="", for_fit=False, path=""):
        self.forfit = for_fit
        if not for_fit:
            self.__voigt = m
            self.__tensor = self.__matrix2comp(self.__voigt)
            self.__name = descr
            self.__norm = self.__tensornorm(self.__tensor)
            self.__sym = self.__sympart(self.__tensor)
            self.__skw = self.__residualpart(self.__tensor)
            self.__canonictensor = self.__canonicform()
            self.__canonicvoigt = self.__comp2matrix(self.__canonictensor)
            self.__canonicorientations = self.__allorientations(self.__canonictensor)
            self.__path = path
            self.numcanonicorientations = len(self.__canonicorientations)
        else:
            self.__voigt = m
            self.__tensor = np.zeros((3, 3, 3, 3))
            self.__name = descr
            self.__norm = self.__tensornorm(self.__tensor)
            self.__sym = self.__sympart(self.__tensor)
            self.__skw = self.__residualpart(self.__tensor)
            self.__canonictensor = self.__canonicform()
            self.__canonicvoigt = self.__comp2matrix(self.__canonictensor)
            self.__canonicorientations = self.__allorientations(self.__canonictensor)
            self.__path = path
            self.numcanonicorientations = len(self.__canonicorientations)


    def __add__(self, other):
        return Stiffness(self.__voigt + other.getVoigt(), "Sum of " + self.__name + " and " + other.getName())

    def __sub__(self, other):
        return Stiffness(self.__voigt - other.getVoigt(), "Difference of " + self.__name + " and " + other.getName())

    def __mul__(self, other):
        if isinstance(other, Stiffness):
            A = self.__tensor
            B = other.getTetrade()
            prod = 0
            for i, j, k, l in it.product(range(3), repeat=4):
                prod += A[i, j, k, l] * B[i, j, k, l]
            return prod
        elif isinstance(other, float) or isinstance(other, int):
            return Stiffness(other * self.__voigt, descr=self.__name)
        else:
            raise TypeError
            return None

    def __truediv__(self, other):
        assert isinstance(other, int) or isinstance(other, float)
        return Stiffness(self.__voigt / other, self.__name)

    def __eq__(self, other):
        assert isinstance(other, Stiffness)
        return self.__tensornorm(self.__tensor - other.getTetrade()) <= 10 ** (-10)

    def __ne__(self, other):
        assert isinstance(other, Stiffness)
        return self.__tensornorm(self.__tensor - other.getTetrade()) > 10 ** (-10)

    def __str__(self):
        if self.forfit:
            return str(self.__voigt)
        else:
            return str(np.round(self.__voigt, decimals=3))


    @classmethod
    def fromFile(cls, fname, descr=None, pat1=r'\{*([-]?\d+\.?\d*)(\*10)\^(\((-)?[0]?(\d+)\))\}*',
                 pat2=r'\{*([-]?\d+\.?\d*)\}*', subpat1=r'\1e\4\5', subpat2=r'\1'):
        """
        Class method to read Voigt-representation from file
        :param fname: Filename of input file
        :type fname: str
        :param descr: Descriptor of the stiffness
        :type descr: str
        :param pat1: RegEx pattern for file input
        :type pat1: str
        :param pat2: RegEx pattern for file input
        :type pat2: str
        :param subpat1: RegEx substitution pattern for pat1
        :type subpat1: str
        :param subpat2: RegEx substitution pattern for pat2
        :type subpat2: str
        :return: None
        """
        with open(fname, 'r') as file:
            f = file.read()
            regex1 = re.compile(pat1)
            regex2 = re.compile(pat2)
            f = regex1.sub(subpat1, f)
            f = regex2.sub(subpat2, f)
            f = f.split(',')
            for i, string in enumerate(f):
                f[i] = float(string)
            m = np.array(f)
            m = m.reshape((6, 6))
        if descr is None:
            if re.search(r'K\d+', fname):
                descr = re.findall(r'K?\d{2,3}(?:fit)?', fname)[0]
            else:
                defor = re.findall(r'Def\d', fname)[0]
                shearnum = re.findall(r'\d{2,3}', fname)[0]
                descr = 'K' + shearnum + ' ' + defor
        return cls(m, descr, path=fname)

    @classmethod
    def fromTensor(cls, t, descr):
        return cls(cls.__comp2matrix(cls, t), descr)

    @classmethod
    def fromFitData(cls, fdata, descr=None):
        m = fdata.getFitResult()
        if descr is None:        
            descr = fdata.getName()
        return cls(m, descr)

    @classmethod
    def randomTriclinic(cls, scale=10000, scatt=False, scattscale=0.05, descr="RandomTriclinic"):
        l = []
        for i in range(21):
            if i in [3, 4, 5, 8, 9, 10, 12, 13, 14]:
                mult = np.sqrt(2)
            elif i in [15, 16, 17, 18, 19, 20]:
                mult = 2
            else:
                mult = 1
            l.append(mult * random())

        m = scale * cls.__symmat6x6(cls, l)
        if scatt:
            mscatt = scattscale * scale * cls.__symmat6x6(cls, [random() for i in range(21)])
            mscatt = (mscatt + np.transpose(mscatt)) / 2
            m = m + mscatt

        return cls(m, descr)

    @classmethod
    def randomMonoclinic(cls, scale=10000, scatt=False, scattscale=0.05, descr="RandomMonolinic"):
        l = [0 for i in range(21)]
        for i in [0, 1, 2, 4, 6, 7, 9, 11, 13, 15, 17, 18, 20]:
            if i in [3, 4, 5, 8, 9, 10, 12, 13, 14]:
                mult = np.sqrt(2)
            elif i in [15, 16, 17, 18, 19, 20]:
                mult = 2
            else:
                mult = 1
            l[i] = mult * random()

        m = scale * cls.__symmat6x6(cls, l)
        if scatt:
            mscatt = scattscale * scale * cls.__symmat6x6(cls, [random() for i in range(21)])
            mscatt = (mscatt + np.transpose(mscatt)) / 2
            m = m + mscatt
        return cls(m, descr)

    @classmethod
    def randomOrthotropic(cls, scale=10000, scatt=False, scattscale=0.05, descr="RandomOrthotropic"):
        l = [0 for i in range(21)]
        for i in [0, 1, 2, 6, 7, 11, 15, 18, 20]:
            if i in [3, 4, 5, 8, 9, 10, 12, 13, 14]:
                mult = np.sqrt(2)
            elif i in [15, 16, 17, 18, 19, 20]:
                mult = 2
            else:
                mult = 1
            l[i] = mult * random()

        m = scale * cls.__symmat6x6(cls, l)
        if scatt:
            mscatt = scattscale * scale * cls.__symmat6x6(cls, [random() for i in range(21)])
            mscatt = (mscatt + np.transpose(mscatt)) / 2
            m = m + mscatt
        return cls(m, descr)

    @classmethod
    def randomCubic(cls, scale=10000, scatt=False, scattscale=0.05, descr="RandomCubic"):
        l = [0 for i in range(21)]
        for i in [0, 1, 15]:
            if i in [3, 4, 5, 8, 9, 10, 12, 13, 14]:
                mult = np.sqrt(2)
            elif i in [15, 16, 17, 18, 19, 20]:
                mult = 2
            else:
                mult = 1
            l[i] = mult * random()
        l[6] = l[0]
        l[11] = l[0]
        l[2] = l[1]
        l[7] = l[1]
        l[18] = l[15]
        l[20] = l[15]
        m = scale * cls.__symmat6x6(cls, l)
        if scatt:
            mscatt = scattscale * scale * cls.__symmat6x6(cls, [random() for i in range(21)])
            mscatt = (mscatt + np.transpose(mscatt)) / 2
            m = m + mscatt
        return cls(m, descr)

    @classmethod
    def randomTrigonal(cls, scale=10000, scatt=False, scattscale=0.05, descr="RandomTrigonal"):
        l = [0 for i in range(21)]
        for i in [0, 1, 2, 3, 9, 11, 15]:
            if i in [3, 4, 5, 8, 9, 10, 12, 13, 14]:
                mult = np.sqrt(2)
            elif i in [15, 16, 17, 18, 19, 20]:
                mult = 2
            else:
                mult = 1
            l[i] = mult * random()
        l[6] = l[0]
        l[7] = l[2]
        l[8] = -l[3]
        l[4] = -l[9]
        l[18] = l[15]
        l[20] = l[0] - l[1]
        l[17] = 2 * l[9] / np.sqrt(2)
        l[19] = - 2 * l[8] / np.sqrt(2)
        m = scale * cls.__symmat6x6(cls, l)
        if scatt:
            mscatt = scattscale * scale * cls.__symmat6x6(cls, [random() for i in range(21)])
            mscatt = (mscatt + np.transpose(mscatt)) / 2
            m = m + mscatt
        return cls(m, descr)

    @classmethod
    def randomTetragonal(cls, scale=10000, scatt=False, scattscale=0.05, descr="RandomTetragonal"):
        l = [0 for i in range(21)]
        for i in [0, 1, 2, 5, 11, 15, 20]:
            if i in [3, 4, 5, 8, 9, 10, 12, 13, 14]:
                mult = np.sqrt(2)
            elif i in [15, 16, 17, 18, 19, 20]:
                mult = 2
            else:
                mult = 1
            l[i] = mult * random()
        l[6] = l[0]
        l[7] = l[2]
        l[10] = - l[5]
        l[18] = l[15]
        m = scale * cls.__symmat6x6(cls, l)
        if scatt:
            mscatt = scattscale * scale * cls.__symmat6x6(cls, [random() for i in range(21)])
            mscatt = (mscatt + np.transpose(mscatt)) / 2
            m = m + mscatt
        return cls(m, descr)

    @classmethod
    def randomHexagonal(cls, scale=10000, scatt=False, scattscale=0.05, descr="RandomHexagonal"):
        l = [0 for i in range(21)]
        for i in [0, 1, 2, 11, 15]:
            if i in [3, 4, 5, 8, 9, 10, 12, 13, 14]:
                mult = np.sqrt(2)
            elif i in [15, 16, 17, 18, 19, 20]:
                mult = 2
            else:
                mult = 1
            l[i] = mult * random()
        l[6] = l[0]
        l[7] = l[2]
        l[18] = l[15]
        l[20] = l[0] - l[1]
        m = scale * cls.__symmat6x6(cls, l)
        if scatt:
            mscatt = scattscale * scale * cls.__symmat6x6(cls, [random() for i in range(21)])
            mscatt = (mscatt + np.transpose(mscatt)) / 2
            m = m + mscatt
        return cls(m, descr)

    @classmethod
    def randomIsotropic(cls, scale=10000, scatt=False, scattscale=0.05, descr="RandomIsotropic"):
        l = [0 for i in range(21)]
        for i in [0, 1]:
            if i in [3, 4, 5, 8, 9, 10, 12, 13, 14]:
                mult = np.sqrt(2)
            elif i in [15, 16, 17, 18, 19, 20]:
                mult = 2
            else:
                mult = 1
            l[i] = mult * random()
        l[6] = l[0]
        l[11] = l[0]
        l[2] = l[1]
        l[7] = l[1]
        l[15] = l[0] - l[1]
        l[18] = l[0] - l[1]
        l[20] = l[0] - l[1]
        m = scale * cls.__symmat6x6(cls, l)
        if scatt:
            mscatt = scattscale * scale * cls.__symmat6x6(cls, [random() for i in range(21)])
            mscatt = (mscatt + np.transpose(mscatt)) / 2
            m = m + mscatt
        return cls(m, descr)

    def __symmat6x6(self, l):
        return np.array([[l[0], l[1], l[2], l[3], l[4], l[5]],
                         [l[1], l[6], l[7], l[8], l[9], l[10]],
                         [l[2], l[7], l[11], l[12], l[13], l[14]],
                         [l[3], l[8], l[12], l[15], l[16], l[17]],
                         [l[4], l[9], l[13], l[16], l[18], l[19]],
                         [l[5], l[10], l[14], l[17], l[19], l[20]]])

    def distanceTo(self, obj, normed=True, canonic=False):
        """
        Calculates distance between instance and obj
        :param obj: SymmetryGroup or Stiffness instance
        :param normed: (opitional, default True) if True returns normed distance between to stiffness instances
        :return: Distance to symmetry group or distance
        """
        assert isinstance(obj, SymmetryGroup) or isinstance(obj, Stiffness)
        if isinstance(obj, SymmetryGroup):
            tempdistlist = []
            for orientation in self.__canonicorientations:
                orientation = orientation / self.__norm
                if obj.getName() != "isotropic":
                    projectedstiffness = np.zeros((3, 3, 3, 3))
                    for trafo in obj.getElements():
                        projectedstiffness += self.__rayleighprod(trafo, orientation)
                    projectedstiffness = projectedstiffness / obj.numelem
                else:
                    projectedstiffness = self.__isopart(orientation)
                tempdistlist.append(self.__tensornorm(orientation - projectedstiffness))
            return np.min(tempdistlist)
        elif isinstance(obj, Stiffness):
            if canonic:
                if normed:                                                        
                    return np.min([self.__tensornorm(A - B) / self.__norm for A, B in it.product(self.__canonicorientations, obj.getCanonicOrientations())])
                else:
                    return self.__tensornorm(self.__canonictensor - obj.getCanonicTetrade())
            else:
                if normed:
                    return self.__tensornorm(self.__tensor - obj.getTetrade()) / self.__norm 
                else:
                    return self.__tensornorm(self.__tensor - obj.getTetrade())

    def relativeError(self, obj):
        assert isinstance(obj, Stiffness)
        return self.__tensornorm(self.__tensor - obj.getTetrade()) / self.__norm

    def fitWith(self, fitstiff, pat=r'\w\d{4}(?=\Z)', writeoutput=True, path="output/", canonic=False, tol=-12, coeffs=None):
        if canonic:
            diff = fitstiff.getVoigt() - self.__canonicvoigt
        else:
            diff = fitstiff.getVoigt() - self.__voigt
        diff = diff.reshape((1, 36))

        if coeffs is None:
            coeffset = set(fitstiff.getVoigt().flatten())
            try:
                coeffset.remove(0)
            except KeyError:
                print("Warning: no 0 in coefficient set")


            coeffset = [str(i) for i in coeffset]



            tmpcoeffs = []
            for elem in coeffset:
                tmpcoeffs += re.split("\+|-|\*|\s+|e|\.|/", elem)

            regex = re.compile(pat)

            coeffset = []
            for i in tmpcoeffs:
                if regex.match(i) is not None:
                    coeffset.append(i)

            coeffset = set(coeffset)
            coeffset = [Symbol(i, Real=True) for i in coeffset]
        else:
            coeffset = list(coeffs);


        l = []
        for i in range(36):
            l.append(poly(diff[0, i], coeffset).as_dict())

        t = tuple([0 for p in range(len(coeffset))])
        for i, elem in enumerate(l):
            if not t in elem.keys():
                tmpdict = elem
                tmpdict[t] = 0
                l[i] = tmpdict

        A = []
        b = []

        for ind, elem in enumerate(l):
            ALine = []
            for key, val in elem.items():
                if ALine == []:
                    ALine = [0 for i in range(len(key))]
                if set(key) == {0}:
                    b.append(-val)
                for keyind, keyelem in enumerate(key):
                    if keyelem != 0:
                        ALine[keyind] += val
            A.append(ALine)
        A = np.asarray(A)
        b = np.asarray(b)

        A = A.astype(float)
        b = b.astype(float)

        res = np.linalg.lstsq(A, b, rcond=-1)

        if writeoutput:
            data = FitData("fit_" + self.__name.replace(' ','_') + ".txt", path)
            for i, key in enumerate(coeffset):
                data[key] = res[0][i]
            data["FitMatrix"] = self.__matrixstring(fitstiff.getVoigt(), clean=False, tol=tol)
            data["FitMatrixClean"] = self.__matrixstring(fitstiff.getVoigt(), clean=True, tol=tol)
            return None

        return res, coeffset

    def closestIn(self, group):
        """
        Returns the closest stiffness in a given symmetry class
        :param group: Symmetry group associated with symmetry class
        :type group: SymmetryGroup
        :return: Closest stiffness in group (Stiffness object)
        """
        assert isinstance(group, SymmetryGroup)
        tempdistlist = []
        projectedstiffnesslist = []
        for orientation in self.__canonicorientations:
            if group.getName() != "isotropic":
                projectedstiffness = np.zeros((3, 3, 3, 3))
                for trafo in group.getElements():
                    projectedstiffness += self.__rayleighprod(trafo, orientation)
                projectedstiffness = projectedstiffness / group.numelem
            else:
                projectedstiffness = self.__isopart(orientation)
            projectedstiffnesslist.append(projectedstiffness)
            tempdistlist.append(self.__tensornorm(self.__tensor - projectedstiffness))
        return Stiffness.fromTensor(projectedstiffnesslist[tempdistlist.index(min(tempdistlist))],
                                    "Closest to " + self.__name + " in " + group.getName())

    def anisotropyration(self):
        return self.__tensornorm(self.__anipart()) / self.__tensornorm(self.__isopart())

    def __matrixstring(self, out, clean, pat1=r'(\d+\.\d*e(-?0?\d+))\*?\w*', pat2=r'(?<!\d\.)0\*\w\d{1,4}', pat3=r'(\w[1-9]{1,4})0*',
                       pat4=r'(\[|,)-?0{2,999}', pat5=r'(\[|,)0\+', tol=-12):
        s = '['
        for i, j in it.product(range(len(out[0])), repeat=2):
            if j == min(range(len(out[0]))):
                s += '['
            s += str(out[i, j])
            if j != max(range(len(out[0]))):
                s += ','
            elif i != max(range(len(out[0]))):
                s += '],'
            else:
                s += ']'
        s += ']'
        if clean:
            regex = re.compile(pat1)
            pairs = regex.findall(s)
            zerolist = []
            for item in pairs:
                if int(item[1]) < tol:
                    zerolist.append(item[0])
            for zero in zerolist:
                s = re.sub(zero, '0', s)

            # print(s)

            regex = re.compile(pat2)
            s = regex.sub(r'0', s)
            # print(s)
            s = re.sub(r'(\+ 0|- 0)(?!\.)', r'0', s)
            # print(s)
            s = re.sub(r'\s', r'', s)
            # print(s)


            regex = re.compile(pat3)
            s = regex.sub(r'\1', s)
            # print(s)

            regex = re.compile(pat4)
            s = regex.sub(r'\g<1>0', s)
            # print(s)

            regex = re.compile(pat5)
            s = regex.sub(r'\1', s)
            # print(s)
        return s

    def __isopart(self, K=None):
        if K is None:
            K = self.__tensor
        elist = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        C = np.zeros((3, 3))
        V = np.zeros((3, 3))
        iso = np.zeros((3, 3, 3, 3))

        for i, j in it.product(range(3), repeat=2):
            d = np.tensordot(elist[i], elist[j], 0)
            Ctemp = 0
            Vtemp = 0
            for k in range(3):
                Ctemp += K[i, j, k, k]
                Vtemp += K[i, k, j, k]
            C += Ctemp * d
            V += Vtemp * d

        trC = 0
        trV = 0
        for i in range(3):
            trC += C[i, i]
            trV += V[i, i]

        H = (trC + 2 * trV) / 45
        h = (trC - trV) / 9

        for i, j, k, l in it.product(range(3), repeat=4):
            d1 = np.tensordot(elist[i], elist[j], 0)
            d2 = np.tensordot(elist[k], elist[l], 0)
            iso += (H * (self.__kroneckerdelta(i, j) * self.__kroneckerdelta(k, l) + self.__kroneckerdelta(i,
                                                                                                           k) * self.__kroneckerdelta(
                j, l)
                         + self.__kroneckerdelta(i, l) * self.__kroneckerdelta(j, k))
                    + h * (self.__kroneckerdelta(i, j) * self.__kroneckerdelta(k, l) - self.__kroneckerdelta(i,
                                                                                                             k) * self.__kroneckerdelta(
                        j, l) / 2
                           - self.__kroneckerdelta(i, l) * self.__kroneckerdelta(j, k) / 2)) * np.tensordot(d1, d2, 0)
        return iso

    def __anipart(self):
        return self.__tensor - self.__isopart()

    def __tensornorm(self, K):
        n = 0
        for a, b, c, d in it.product(range(3), repeat=4):
            n += K[a, b, c, d] ** 2
        return np.sqrt(n)

    def __sympart(self, K):
        elist = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        res = np.zeros((3, 3, 3, 3))
        for i, j, k, l in it.product(range(3), range(3), range(3), range(3)):
            d1 = np.tensordot(elist[i], elist[j], 0)
            d2 = np.tensordot(elist[k], elist[l], 0)
            res += (1 / 3) * (K[i, j, k, l] + K[i, k, l, j] + K[i, l, j, k]) * np.tensordot(d1, d2, 0)
        return res

    def __residualpart(self, k):
        return k - self.__sympart(k)

    def __kroneckerdelta(self, i, j):
        if i == j:
            return 1
        else:
            return 0

    def __baerheimt(self, A):
        elist = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        c1 = np.zeros((3, 3))
        c2 = 0

        for i, j in it.product(range(3), range(3)):
            c2 += A[i, i, j, j]
            for p in range(3):
                c1[i, j] += A[i, j, p, p]
        res = np.zeros((3, 3))
        for i, j in it.product(range(3), range(3)):
            res += (c1[i, j] - (c2 / 4) * self.__kroneckerdelta(i, j)) * np.tensordot(elist[i], elist[j], 0)

        return res

    def __canonictransform(self):
        t = self.__baerheimt(self.__skw)
        eigvalues, eigvecs = np.linalg.eig(t)
        for i in range(len(eigvecs)):
            eigvecs[i] = eigvecs[i] / np.linalg.norm(eigvecs[i])
        return eigvecs

    def __canonicform(self):
        return self.__rayleighprod(np.transpose(self.__canonictransform()), self.__tensor)

    def __allorientations(self, k):
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        orientchanges = [self.__rotationmatrix(np.pi / 2, e1),
                         self.__rotationmatrix(-np.pi / 2, e1),
                         self.__rotationmatrix(np.pi, e1),
                         self.__rotationmatrix(np.pi / 2, e2),
                         self.__rotationmatrix(-np.pi / 2, e2),
                         self.__rotationmatrix(np.pi, e2),
                         self.__rotationmatrix(np.pi / 2, e3),
                         self.__rotationmatrix(-np.pi / 2, e3),
                         self.__rotationmatrix(np.pi, e3),
                         self.__rotationmatrix(2 * np.pi / 3, e1 + e2 + e3),
                         self.__rotationmatrix(4 * np.pi / 3, e1 + e2 + e3),
                         np.identity(3)]

        return [self.__rayleighprod(orientchanges[i], k) for i in range(len(orientchanges))]

    def __rotationmatrix(self, theta, u):
        u = u / np.linalg.norm(u)
        return np.array(
            [[np.cos(theta) + u[0] ** 2 * (1 - np.cos(theta)), u[0] * u[1] * (1 - np.cos(theta)) - u[2] * np.sin(theta),
              u[0] * u[2] * (1 - np.cos(theta)) + u[1] * np.sin(theta)],
             [u[0] * u[1] * (1 - np.cos(theta)) + u[2] * np.sin(theta), np.cos(theta) + u[1] ** 2 * (1 - np.cos(theta)),
              u[1] * u[2] * (1 - np.cos(theta)) - u[0] * np.sin(theta)],
             [u[0] * u[2] * (1 - np.cos(theta)) - u[1] * np.sin(theta),
              u[1] * u[2] * (1 - np.cos(theta)) + u[0] * np.sin(theta),
              np.cos(theta) + u[2] ** 2 * (1 - np.cos(theta))]])

    def __rayleighprod(self, A, B):
        elist = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        prod = np.zeros((3, 3, 3, 3))
        for a, b, c, d in it.product(range(3), range(3), range(3), range(3)):
            d1 = np.tensordot(elist[a], elist[b], 0)
            d2 = np.tensordot(elist[c], elist[d], 0)
            comp = 0
            for i, j, k, l in it.product(range(3), range(3), range(3), range(3)):
                comp += A[a, i] * A[b, j] * A[c, k] * A[d, l] * B[i, j, k, l]
            prod += comp * np.tensordot(d1, d2, 0)
        return prod

    def __matrix2comp(self, m):
        comp = np.zeros((3, 3, 3, 3))
        comp[0, 0, 0, 0] = m[0, 0]
        comp[0, 0, 1, 1] = m[0, 1]
        comp[0, 0, 2, 2] = m[0, 2]
        comp[0, 0, 1, 2] = m[0, 3] / np.sqrt(2)
        comp[0, 0, 0, 2] = m[0, 4] / np.sqrt(2)
        comp[0, 0, 0, 1] = m[0, 5] / np.sqrt(2)
        comp[1, 1, 1, 1] = m[1, 1]
        comp[1, 1, 2, 2] = m[1, 2]
        comp[1, 1, 1, 2] = m[1, 3] / np.sqrt(2)
        comp[0, 2, 1, 1] = m[1, 4] / np.sqrt(2)
        comp[0, 1, 1, 1] = m[1, 5] / np.sqrt(2)
        comp[2, 2, 2, 2] = m[2, 2]
        comp[1, 2, 2, 2] = m[2, 3] / np.sqrt(2)
        comp[0, 2, 2, 2] = m[2, 4] / np.sqrt(2)
        comp[0, 1, 2, 2] = m[2, 5] / np.sqrt(2)
        comp[1, 2, 1, 2] = m[3, 3] / 2
        comp[0, 2, 1, 2] = m[3, 4] / 2
        comp[0, 1, 1, 2] = m[3, 5] / 2
        comp[0, 2, 0, 2] = m[4, 4] / 2
        comp[0, 1, 0, 2] = m[4, 5] / 2
        comp[0, 1, 0, 1] = m[5, 5] / 2

        for i, j, k, l in it.product(range(3), range(3), range(3), range(3)):
            comp[j, i, k, l] = comp[i, j, k, l]
            comp[i, j, l, k] = comp[i, j, k, l]
            comp[k, l, i, j] = comp[i, j, k, l]

        return comp

    def __comp2matrix(self, k):
        m0 = np.array(
            [k[0, 0, 0, 0], k[0, 0, 1, 1], k[0, 0, 2, 2], np.sqrt(2) * k[0, 0, 1, 2], np.sqrt(2) * k[0, 0, 2, 0],
             np.sqrt(2) * k[0, 0, 0, 1]])
        m1 = np.array(
            [k[1, 1, 0, 0], k[1, 1, 1, 1], k[1, 1, 2, 2], np.sqrt(2) * k[1, 1, 1, 2], np.sqrt(2) * k[1, 1, 2, 0],
             np.sqrt(2) * k[1, 1, 0, 1]])
        m2 = np.array(
            [k[2, 2, 0, 0], k[2, 2, 1, 1], k[2, 2, 2, 2], np.sqrt(2) * k[2, 2, 1, 2], np.sqrt(2) * k[2, 2, 2, 0],
             np.sqrt(2) * k[2, 2, 0, 1]])
        m3 = np.array(
            [np.sqrt(2) * k[1, 2, 0, 0], np.sqrt(2) * k[1, 2, 1, 1], np.sqrt(2) * k[1, 2, 2, 2], 2 * k[1, 2, 1, 2],
             2 * k[1, 2, 2, 0], 2 * k[1, 2, 0, 1]])
        m4 = np.array(
            [np.sqrt(2) * k[2, 0, 0, 0], np.sqrt(2) * k[2, 0, 1, 1], np.sqrt(2) * k[2, 0, 2, 2], 2 * k[2, 0, 1, 2],
             2 * k[2, 0, 2, 0], 2 * k[2, 0, 0, 1]])
        m5 = np.array(
            [np.sqrt(2) * k[0, 1, 0, 0], np.sqrt(2) * k[0, 1, 1, 1], np.sqrt(2) * k[0, 1, 2, 2], 2 * k[0, 1, 1, 2],
             2 * k[0, 1, 2, 0], 2 * k[0, 1, 0, 1]])
        return np.array([m0, m1, m2, m3, m4, m5])

    def getVoigt(self):
        return self.__voigt

    def getTetrade(self):
        return self.__tensor

    def getName(self):
        return self.__name

    def getCanonicOrientations(self):
        return self.__canonicorientations

    def norm(self):
        return self.__norm

    def getSymPart(self):
        return self.__sym

    def getResidualPart(self):
        return self.__skw

    def getCanonicTetrade(self):
        return self.__canonictensor

    def getCanonicVoigt(self):
        return self.__canonicvoigt

    def getIsotropicPart(self):
        return self.__isopart(self.__tensor)

    def getAnisotropicPart(self):
        return self.__anipart(self.__tensor)

    def getpath(self):
        return self.__path
