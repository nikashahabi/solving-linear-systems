import numpy as np
import time

def iterativeMethods(A, M, b, stop):
    # start = time.time()
    'general iterative method to solve AX=b: M is the splitting matrix and is nonsingular. A = M-N.'
    'X(k+1) = Minverse * N * X(k) + Minverse * b'
    assert(A.shape[0] == A.shape[1] == M.shape[0] == M.shape[1])
    n = A.shape[0]
    N = M - A
    print("splitting matrix =")
    print(M)
    X = np.zeros((n, 1))
    print("Initial approximation =")
    print(X)
    Minverse = np.linalg.inv(M)
    MinverseDotb = np.dot(Minverse, b)
    MinverseDotN = np.dot(Minverse, N)
    Xnew = np.dot(MinverseDotN, X) + MinverseDotb
    i = 1
    while 1:
        print("X approximation after %d iterations =" %i)
        print(Xnew)
        print("norm 2 of X{i} - X{k} =".format(i=i, k=i-1))
        print(np.linalg.norm(X - Xnew))
        if np.linalg.norm(X-Xnew) <= stop:
            print("iterative method terminated")
            print("final approximation of X is =")
            print(Xnew)
            # end = time.time()
            # time = end - start
            # print("execution time =")
            # print(time)
            return Xnew, i
        X = np.ndarray.copy(Xnew)
        Xnew = np.dot(MinverseDotN, Xnew) + MinverseDotb
        i += 1


def getDEF(A):
    'returns D, E and F. A = D - E - F'
    assert(A.shape[0] == A.shape[1])
    n = A.shape[0]
    D = np.zeros((n, n))
    E = np.zeros((n, n))
    for i in range(n):
        D[i][i] = A[i][i]
        for j in range(i):
            E[i][j] = -1 * A[i][j]
    F = D - E - A
    return D, E, F


def Jacobi(A, b, stop):
    'implements Jacobi algorithm with general iterative method'
    print("jacobi's algorithm matrix implementation:")
    D,E,F = getDEF(A)
    M = D
    return iterativeMethods(A, M, b, stop)


def JacobiV2(A, b, stop):
    'jmplements jacobi algorithm'
    print("jacobi's algorithm:")
    assert (A.shape[0] == A.shape[1])
    n = A.shape[0]
    X = np.zeros((n, 1))
    Xtemp = np.zeros((n, 1))
    Xnew = np.zeros((n, 1))
    print("Initial approximation =")
    print(X)
    for i in range(n):
        Xnew[i][0] = b[i][0]
        for j in range(n):
            if j != i:
                Xnew[i][0] -= A[i][j] * X[j][0]
        Xnew[i][0] *= 1 / A[i][i]
    counter = 1
    while 1:
        print("X approximation after %d iterations =" % counter)
        print(Xnew)
        print("norm 2 of X{i} - X{k} =".format(i=counter, k=counter - 1))
        print(np.linalg.norm(X - Xnew))
        if np.linalg.norm(X - Xnew) <= stop:
            print("iterative method terminated")
            print("final approximation of X is =")
            print(Xnew)
            # end = time.time()
            # time = end - start
            # print("execution time =")
            # print(time)
            return Xnew, counter
        X = np.ndarray.copy(Xnew)
        for i in range(n):
            Xtemp[i][0] = b[i][0]
            for j in range(n):
                if j != i:
                    Xtemp[i][0] -= A[i][j] * Xnew[j][0]
            Xtemp[i][0] *= 1/A[i][i]
        Xnew = np.ndarray.copy(Xtemp)
        counter += 1




def GaussSidel(A, b, stop):
    'implements GS algorithm with general iterative method'
    print("Gauss-Sidel's algorithm matrix implementation:")
    D, E, F = getDEF(A)
    M = D - E
    return iterativeMethods(A, M, b, stop)


def GaussSidelV2(A, b, stop):
    'implements GS algorithm'
    print("Gauss-Sidel's algorithm:")
    assert (A.shape[0] == A.shape[1])
    n = A.shape[0]
    X = np.zeros((n, 1))
    Xnew = np.ndarray.copy(X)
    print("Initial approximation =")
    print(X)
    for i in range(n):
        Xnew[i][0] = b[i][0]
        for j in range(n):
            if j != i:
                Xnew[i][0] -= A[i][j] * Xnew[j][0]
        Xnew[i][0] *= 1 / A[i][i]
    counter = 1
    while 1:
        print("X approximation after %d iterations =" % counter)
        print(Xnew)
        print("norm 2 of X{i} - X{k} =".format(i=counter, k=counter - 1))
        print(np.linalg.norm(X - Xnew))
        if np.linalg.norm(X - Xnew) <= stop:
            print("iterative method terminated")
            print("final approximation of X is =")
            print(Xnew)
            # end = time.time()
            # time = end - start
            # print("execution time =")
            # print(time)
            return Xnew, counter
        X = np.ndarray.copy(Xnew)
        for i in range(n):
            Xnew[i][0] = b[i][0]
            for j in range(n):
                if j != i:
                    Xnew[i][0] -= A[i][j] * Xnew[j][0]
            Xnew[i][0] *= 1 / A[i][i]
        counter += 1


def SOR(A, b, stop, omega):
    'implements SOR algorithm with general iterative method'
    if omega <= 0 or omega >= 2:
        print("SOR will not be convergent with this omega")
        return
    print("SOR algorithm matrix implementation with omega = :", omega)
    D, E, F = getDEF(A)
    M = 1/omega * (D - E)
    return iterativeMethods(A, M, b, stop)


def SORV2(A, b, stop, omega):
    'implements SOR'
    if omega <= 0 or omega >= 2:
        print("SOR will not be convergent with this omega")
        return
    print("SOR algorithm:")
    assert (A.shape[0] == A.shape[1])
    n = A.shape[0]
    X = np.zeros((n, 1))
    Xnew = np.ndarray.copy(X)
    print("Initial approximation =")
    print(X)
    for i in range(n):
        temp = b[i][0]
        for j in range(n):
            if j != i:
                temp -= A[i][j] * Xnew[j][0]
        temp *= 1 / A[i][i]
        delta = temp - Xnew[i][0]
        Xnew[i][0] += omega * delta
    counter = 1
    while 1:
        print("X approximation after %d iterations =" % counter)
        print(Xnew)
        print("norm 2 of X{i} - X{k} =".format(i=counter, k=counter - 1))
        print(np.linalg.norm(X - Xnew))
        if np.linalg.norm(X - Xnew) <= stop:
            print("iterative method terminated")
            print("final approximation of X is =")
            print(Xnew)
            # end = time.time()
            # time = end - start
            # print("execution time =")
            # print(time)
            return Xnew, counter
        X = np.ndarray.copy(Xnew)
        for i in range(n):
            temp = b[i][0]
            for j in range(n):
                if j != i:
                    temp -= A[i][j] * Xnew[j][0]
            temp *= 1 / A[i][i]
            delta = temp - Xnew[i][0]
            Xnew[i][0] += omega * delta
        counter += 1


def getTridiagonal(a, b, c, n):
    'returns a tridiagonal  n*n matrix. bellow diagonal a, on diagonal b and above diagonal c'
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i][i] = b
        if 0 <= i-1 < n:
            matrix[i-1][i] = a
        if 0 <= i+1 < n:
            matrix[i+1][i] = c
    return(matrix)


def LUWithPivoting(A):
    'saves the LU decomposition of A in A and the permutation in intch'
    'flag is False if A is singular and PtLu decomposition does not exist'
    'returns flag and intch'
    assert(A.shape[0] == A.shape[1])
    n = A.shape[0]
    intch = np.zeros((n, 1))
    flag = True
    for k in range(n-1):
        amax = abs(A[k][k])
        amaxIndex = k
        for i in range(k, n, 1):
            if abs(A[i][k]) > abs(amax):
                amax = abs(A[i][k])
                amaxIndex = i
        if (amax == 0):
            flag = False
            intch[k] = 0
        else:
            intch[k] = amaxIndex
            if (amaxIndex != k):
                A[[amaxIndex, k]] = A[[k, amaxIndex]]
                print("pivoting")
                # print(A)
            for i in range(k+1, n, 1):
                A[i][k] = A[i][k] / A[k][k]

            for i in range(k+1, n, 1):
                for x in range(k+1, n, 1):
                    A[i][x] = A[i][x] - A[i][k] * A[k][x]
            # print("A after first iteration")
            # print(A)
    if A[n-1][n-1] == 0:
        flag = False
        intch[n-1] = 0
    else:
        intch[n-1] = n-1
    if flag is True:
        print("A is nonsigular")
    else:
        print("A is singular")
    print("LU decomposition of Ahat(Permutation * A)")
    print(A)
    print("Permutation")
    print(intch)
    return flag, intch


def LUWithoutPivoting(A):
    'saves the LU decomposition of A in A.'
    'flag is True if all of A leading principal submatrices are nonsingular'
    'and LU decomposition exists'
    assert (A.shape[0] == A.shape[1])
    n = A.shape[0]
    flag = True
    for k in range(n - 1):
        for i in range(k + 1, n, 1):
            if A[k][k] == 0:
                flag = False
                break
            A[i][k] = A[i][k] / A[k][k]
        if flag is False:
            break
        for i in range(k + 1, n, 1):
            for x in range(k + 1, n, 1):
                A[i][x] = A[i][x] - A[i][k] * A[k][x]
    if A[n - 1][n - 1] == 0:
        flag = False
    if flag is False:
        print("one of A's leading principal submatrices is singular and LU decoposition was not found")
    else:
        print("all of A's leading principal submatrices are nonsingular and LU decoposition was found")
        print("LU decomposition of A")
        print(A)
    return flag


def GaussianElimination(A, b):
    'solves AX=b with Gaussian elimination. at the end of this method A holds the LU decomposition'
    'and b is X. returns b'
    assert (A.shape[0] == A.shape[1])
    n = A.shape[0]
    flag, intch = LUWithPivoting(A)
    if flag is False:
        print("A is singular and the linear system can not be solved")
        return
    for k in range(n-1):
        m = intch[k]
        temp = b[k]
        b[k] = b[int(m)]
        b[int(m)] = temp
    print("b after permutation")
    print(b)
    for j in range(n-1):
        for i in range(j+1, n, 1):
            b[i] = b[i] - A[i][j] * b[j]
    print("after solving Ly = b")
    print(b)
    for j in range(n-1, -1, -1):
        assert(A[j][j] != 0)
        b[j] = b[j] / A[j][j]
        for i in range(j):
            b[i] = b[i] - A[i][j] * b[j]
    print("after solving Ux = y")
    print(b)
    return b


# A = getTridiagonal(-1, 4, -1, 4)
# b = np.ones((4, 1))
# iterations = []
# times = []
# stop = 0.0001
#
# start = time.time()
# X, iteration = JacobiV2(A, b, stop)
# end = time.time()
# iterations.append(iteration)
# times.append(end - start)
# start = time.time()
# X, iteration = GaussSidelV2(A, b, stop)
# end = time.time()
# iterations.append(iteration)
# times.append(end - start)
# start = time.time()
# X, iteration = SORV2(A, b, stop, 1.1)
# end = time.time()
# iterations.append(iteration)
# times.append(end - start)
# print(iterations)
# print(times)

# counterList = []
# A = getTridiagonal(-1, 4, -1, 4)
# b = np.ones((4, 1))
# i = 0.2
# while i < 1.6:
#     print(i)
#     X, counter = SOR(A, b, 0.0001, i)
#     counterList.append(counter)
#     i += 2
# X, counter1 = SOR(A, b, 0.0001, 0.9)
# X, counter2 = SOR(A, b, 0.0001, 1.1)
# print(counter1, counter2)
# print(counterList)

# A = getTridiagonal(-1, 4, -1, 16)
# start = time.time()
# LUWithoutPivoting(A)
# end = time.time()
# LUWithPivoting(A)
# end2 = time.time()
# print(end2 - end, end - start)





