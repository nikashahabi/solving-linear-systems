import numpy as np


def iterativeMethods(A, M, b, stop):
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
        print("X approximation after", i, "iterations =")
        print(Xnew)
        print("norm 2 of X", i, "- X", i - 1, " =")
        print(np.linalg.norm(X - Xnew))
        if np.linalg.norm(X-Xnew) <= stop:
            print("iterative method terminated")
            print("final approximation of X is =")
            print(Xnew)
            return Xnew
        X = np.ndarray.copy(Xnew)
        Xnew = np.dot(MinverseDotN, Xnew) + MinverseDotb
        i += 1


def getDEF(A):
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
    print("jacobi's algorithm:")
    D,E,F = getDEF(A)
    M = D
    return iterativeMethods(A, M, b, stop)


def GaussSidel(A, b, stop):
    print("Gauss-Sidel's algorithm:")
    D, E, F = getDEF(A)
    M = D - E
    return iterativeMethods(A, M, b, stop)


def SOR(A, b, stop, omega):
    print("SOR algorithm with omega = :", omega)
    D, E, F = getDEF(A)
    M = 1/omega * (D - E)
    return iterativeMethods(A, M, b, stop)


def tridiagonal(a, b, c, n):
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i][i] = b
        if 0 <= i-1 < n:
            matrix[i-1][i] = a
        if 0 <= i+1 < n:
            matrix[i+1][i] = c
    return(matrix)

A = tridiagonal(-1, 4, -1, 4)
b = np.ones((4, 1))
X = Jacobi(A, b, 0.0001)








