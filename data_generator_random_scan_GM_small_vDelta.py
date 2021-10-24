###############
# INFORMATION #
###############

"""With this .py file you will create a .txt file of randomly generated points from the 7-dimensional parameter space
of the GM model satisfying all theoretical constrains. In this particular file, the independent parameters are taken
to be: M1, M2, tanth, lambda2, lambda3, lambda4, lambda5. We also include in the final file the corresponding values
of all other dependent parameters in the theory."""

###########
# MODULES #
###########

import numpy as np
import cmath
from scipy.interpolate import interp1d
import random as rd
import time

start = time.time()

#############
# CONSTANTS #
#############

v = 246.22
mh = 125.25

##########
# RANGES #
##########

# ---------------------
n_min = -11.0  # vDelta ~ 10e-9
n_max = -2.0  # vDelta ~ 1
# ---------------------
M1coeff_min = -30.0
M1coeff_max = 30.0
# ---------------------
M2coeff_min = -30.0
M2coeff_max = 30.0
# ---------------------
lam2_min = -(2 / 3) * np.pi
lam2_max = (2 / 3) * np.pi
# ---------------------
lam3_min = -(1 / 2) * np.pi
lam3_max = (3 / 5) * np.pi
# ---------------------
lam4_min = -(1 / 5) * np.pi
lam4_max = (1 / 2) * np.pi
# ---------------------
lam5_min = -(8 / 3) * np.pi
lam5_max = (8 / 3) * np.pi
# ---------------------

#############
# FUNCTIONS #
#############

def tanth(n):
    return 1.148 * (10 ** n)


def sinth(n):
    numerator = tanth(n)
    denominator = cmath.sqrt(1 + np.square(tanth(n)))
    return numerator / denominator


def costh(n):
    return cmath.sqrt(1 - np.square(sinth(n)))


def vDelta(n):
    numerator = v * sinth(n)
    denominator = 2 * cmath.sqrt(2)
    return numerator / denominator


def vphi(n):
    numerator = 2 * cmath.sqrt(2) * vDelta(n)
    denominator = tanth(n)
    return numerator / denominator


def m3(n, M1, lam5):
    x = M1 / (4 * vDelta(n))
    y = lam5 / 2
    return cmath.sqrt(np.square(v) * (x + y))


def m5(n, M1, M2, lam3, lam5):
    x = M1 * np.square(vphi(n)) / (4 * vDelta(n))
    y = 12 * M2 * vDelta(n)
    z = (3 / 2) * lam5 * np.square(vphi(n))
    w = 8 * lam3 * np.square(vDelta(n))
    return cmath.sqrt(x + y + z + w)


def M12sq(n, M1, lam2, lam5):
    p = (np.sqrt(3) / 2) * vphi(n)
    x = -M1
    y = 4 * vDelta(n) * (2 * lam2 - lam5)
    return p * (x + y)


def M22sq(n, M1, M2, lam3, lam4):
    x = M1 * np.square(vphi(n)) / (4 * vDelta(n))
    y = -6 * M2 * vDelta(n)
    z = 8 * np.square(vDelta(n)) * (lam3 + 3 * lam4)
    return x + y + z


def lam1(n, M1, M2, lam2, lam3, lam4, lam5):
    p = 1 / (8 * np.square(vphi(n)))
    x = np.square(mh)
    y = np.square(M12sq(n, M1, lam2, lam5)) / (M22sq(n, M1, M2, lam3, lam4) - np.square(mh))
    return p * (x + y)


def M11sq(n, M1, M2, lam2, lam3, lam4, lam5):
    return 8 * lam1(n, M1, M2, lam2, lam3, lam4, lam5) * np.square(vphi(n))


def mu2sq(n, M1, M2, lam2, lam3, lam4, lam5):
    x = -4 * lam1(n, M1, M2, lam2, lam3, lam4, lam5) * np.square(vphi(n))
    y = -3 * (2 * lam2 - lam5) * np.square(vDelta(n))
    z = (3 / 2) * M1 * vDelta(n)
    return x + y + z


def mu3sq(n, M1, M2, lam2, lam3, lam4, lam5):
    x = -(2 * lam2 - lam5) * np.square(vphi(n))
    y = -4 * (lam3 + 3 * lam4) * np.square(vDelta(n))
    z = M1 * np.square(vphi(n)) / (4 * vDelta(n))
    w = 6 * M2 * vDelta(n)
    return x + y + z + w


def mH(n, M1, M2, lam2, lam3, lam4, lam5):
    x = M11sq(n, M1, M2, lam2, lam3, lam4, lam5) + M22sq(n, M1, M2, lam3, lam4)
    y = np.square(mh)
    return cmath.sqrt(x - y)


def GRN(a, b):
    return round(rd.uniform(a, b), 5)


###############
# CONSTRAINTS #
###############

# MASS CONSTRAINTS

def MASS_CSTR_1(n, M1, lam5):
    if np.isreal(m3(n, M1, lam5)) and np.real(m3(n, M1, lam5)) >= 0:
        return True
    else:
        return False


def MASS_CSTR_2(n, M1, M2, lam3, lam5):
    if np.isreal(m5(n, M1, M2, lam3, lam5)) and np.real(m5(n, M1, M2, lam3, lam5)) >= 0:
        return True
    else:
        return False


def MASS_CSTR_3(n, M1, M2, lam2, lam3, lam4, lam5):
    if np.isreal(mH(n, M1, M2, lam2, lam3, lam4, lam5)) and np.real(mH(n, M1, M2, lam2, lam3, lam4, lam5)) >= 0.0 and np.real(mH(n, M1, M2, lam2, lam3, lam4, lam5)) <= 712.0:
        return True
    else:
        return False


# UNITARITY CONSTRAINTS

def UNI_CSTR_1(n, M1, M2, lam2, lam3, lam4, lam5):
    if np.real(cmath.sqrt(np.square(6 * lam1(n, M1, M2, lam2, lam3, lam4, lam5) - 7 * lam3 - 11 * lam4)
                          + 36 * np.square(lam2)) + abs(6 * lam1(n, M1, M2, lam2, lam3, lam4, lam5)
                                                        + 7 * lam3 + 11 * lam4)) < 4 * np.pi:
        return True
    else:
        return False


def UNI_CSTR_2(n, M1, M2, lam2, lam3, lam4, lam5):
    if np.real(cmath.sqrt(np.square(2 * lam1(n, M1, M2, lam2, lam3, lam4, lam5) + lam3 - 2 * lam4)
                          + np.square(lam5)) + abs(2 * lam1(n, M1, M2, lam2, lam3, lam4, lam5)
                                                   - lam3 + 2 * lam4)) < 4 * np.pi:
        return True
    else:
        return False


def UNI_CSTR_3(lam3, lam4):
    if np.real(abs(2 * lam3 + lam4)) < np.pi:
        return True
    else:
        return False


def UNI_CSTR_4(lam2, lam5):
    if np.real(abs(lam2 - lam5)) < 2 * np.pi:
        return True
    else:
        return False


# POTENTIAL BOUNDED FROM BELOW CONSTRAINTS

def BOUND_CSTR_1(n, M1, M2, lam2, lam3, lam4, lam5):
    if lam1(n, M1, M2, lam2, lam3, lam4, lam5) > 0:
        return True
    else:
        return False


def BOUND_CSTR_2(lam3, lam4):
    if any(zeta * lam3 + lam4 < 0 for zeta in np.linspace(1/3, 1, 40)):
        return False
    else:
        return True


def BOUND_CSTR_3(n, M1, M2, lam2, lam3, lam4, lam5):
    if any(lam2 - omega * lam5 + 2 * cmath.sqrt(lam1(n, M1, M2, lam2, lam3, lam4, lam5) * (zeta * lam3 + lam4)) < 0
           for zeta, omega in zip(np.linspace(1/3, 1, 40), np.linspace(-1/4, 1/2, 40))):
        return False
    else:
        return True


####################
# WRITING THE DATA #
####################

N = 12  # number of points

LIST_OF_POINTS = []  # list of independent parameters randomly generated and that satisfy all theoretical constraints

while len(LIST_OF_POINTS) < N:
    L = [
        GRN(n_min, n_max),  # L[0]
        GRN(M1coeff_min, M1coeff_max),  # L[1]
        GRN(M2coeff_min, M2coeff_max),  # L[2]
        GRN(lam2_min, lam2_max),  # L[3]
        GRN(lam3_min, lam3_max),  # L[4]
        GRN(lam4_min, lam4_max),  # L[5]
        GRN(lam5_min, lam5_max)  # L[6]
    ]  # candidate point

    MASS_CSTR = [
        MASS_CSTR_1(L[0], L[1], L[6]),
        MASS_CSTR_2(L[0], L[1], L[2], L[4], L[6]),
        MASS_CSTR_3(L[0], L[1], L[2], L[3], L[4], L[5], L[6])
    ]

    UNI_CSTR = [
        UNI_CSTR_1(L[0], L[1], L[2], L[3], L[4], L[5], L[6]),
        UNI_CSTR_2(L[0], L[1], L[2], L[3], L[4], L[5], L[6]),
        UNI_CSTR_3(L[4], L[5]),
        UNI_CSTR_4(L[3], L[6])
    ]

    BOUND_CSTR = [
        BOUND_CSTR_1(L[0], L[1], L[2], L[3], L[4], L[5], L[6]),
        BOUND_CSTR_2(L[4], L[5]),
        BOUND_CSTR_3(L[0], L[1], L[2], L[3], L[4], L[5], L[6])
    ]

    ALL_CSTR = [
        *MASS_CSTR,
        *UNI_CSTR,
        *BOUND_CSTR
    ]

    if all(ALL_CSTR):
        LIST_OF_POINTS.append([np.real(i) for i in L])
        print("I have found " + str(len(LIST_OF_POINTS)) + " compatible points.")


f = open('otra_tanda.txt', 'w')

s = 15  # white space between the columns in the output .txt file

f.write(
    'tanth'.ljust(s) +
    'M1coeff'.ljust(s) +
    'M2coeff'.ljust(s) +
    'lam1'.ljust(s) +
    'lam2'.ljust(s) +
    'lam3'.ljust(s) +
    'lam4'.ljust(s) +
    'lam5'.ljust(s) +
    'm3'.ljust(s) +
    'm5'.ljust(s) +
    'mH'.ljust(s) +
    'vDelta'.ljust(s) +
    'vphi'.ljust(s) +
    'mu2sq'.ljust(s) +
    'mu3sq'.ljust(s) +
    '\n'
)

for point in LIST_OF_POINTS:
    f.write(
        str(round(np.real(tanth(point[0])), 5)).ljust(s) +  # tanth
        str(point[1]).ljust(s) +  # M1coeff
        str(point[2]).ljust(s) +  # M2coeff
        str(round(np.real(lam1(point[0], point[1], point[2], point[3], point[4], point[5], point[6])), 5)).ljust(s) +  # lam1
        str(point[3]).ljust(s) +  # lam2
        str(point[4]).ljust(s) +  # lam3
        str(point[5]).ljust(s) +  # lam4
        str(point[6]).ljust(s) +  # lam5
        str(round(np.real(m3(point[0], point[1], point[6])), 5)).ljust(s) +  # m3
        str(round(np.real(m5(point[0], point[1], point[2], point[4], point[6])), 5)).ljust(s) +  # m5
        str(round(np.real(mH(point[0], point[1], point[2], point[3], point[4], point[5], point[6])), 5)).ljust(s) +  # mH
        str(round(np.real(vDelta(point[0])), 5)).ljust(s) +  # vDelta
        str(round(np.real(vphi(point[0])), 5)).ljust(s) +  # vphi
        str(round(np.real(mu2sq(point[0], point[1], point[2], point[3], point[4], point[5], point[6])), 5)).ljust(s) +  # mu2sq
        str(round(np.real(mu3sq(point[0], point[1], point[2], point[3], point[4], point[5], point[6])), 5)).ljust(s) + '\n'  # mu3sq
    )

f.close()

#####################
# EXTRA INFORMATION #
#####################

end = time.time()
print("The code execution lasted: " + '\n'
    + str(end - start) + " seconds or " + '\n'
    + str((end - start)/60) + " minutes or" + '\n'
    + str((end - start)/3600) + " hours."
      )  # time taken to run the code

print("Average time to find a compatible point: " + '\n'
    + str((end - start)/len(LIST_OF_POINTS)) + " seconds or " + '\n'
    + str(((end - start)/60)/len(LIST_OF_POINTS)) + " minutes or" + '\n'
    + str(((end - start)/3600)/len(LIST_OF_POINTS)) + " hours."
      )  # time taken to run the code

print("You will find " + str(len(LIST_OF_POINTS)) + " runs in the script.")
