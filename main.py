from concurrent.futures import ProcessPoolExecutor

def matrix_add(A, B):
    """Покомпонентное сложение матриц A и B"""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def matrix_sub(A, B):
    """Покомпонентное вычитание матриц B из A"""
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def split_matrix(A):
    """
    Разбивает матрицу A на 4 подматрицы (A11, A12, A21, A22).
    Предполагается, что размер матрицы — степень двойки.
    """
    n = len(A)
    mid = n // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    return A11, A12, A21, A22

def join_matrices(A11, A12, A21, A22):
    """Объединяет 4 подматрицы в одну"""
    top = [a + b for a, b in zip(A11, A12)]
    bottom = [a + b for a, b in zip(A21, A22)]
    return top + bottom

def classical_multiply(A, B):
    """Классическое умножение матриц"""
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def _strassen(A, B, threshold, depth, max_depth):
    """
    Рекурсивное умножение матриц по алгоритму Штрассена.
    
    Параметры:
      A, B      - входные квадратные матрицы (размер 2^n)
      threshold - порог, при котором используется классическое умножение
      depth     - текущая глубина рекурсии
      max_depth - максимальная глубина параллелизации
    """
    n = len(A)
    if n <= threshold:
        return classical_multiply(A, B)

    # Разбиваем матрицы на 4 подматрицы
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    # Если текущая глубина меньше max_depth, распараллеливаем 7 независимых вызовов
    if depth < max_depth:
        with ProcessPoolExecutor() as pool:
            future1 = pool.submit(_strassen, matrix_add(A11, A22), matrix_add(B11, B22), threshold, depth+1, max_depth)
            future2 = pool.submit(_strassen, matrix_add(A21, A22), B11, threshold, depth+1, max_depth)
            future3 = pool.submit(_strassen, A11, matrix_sub(B12, B22), threshold, depth+1, max_depth)
            future4 = pool.submit(_strassen, A22, matrix_sub(B21, B11), threshold, depth+1, max_depth)
            future5 = pool.submit(_strassen, matrix_add(A11, A12), B22, threshold, depth+1, max_depth)
            future6 = pool.submit(_strassen, matrix_sub(A21, A11), matrix_add(B11, B12), threshold, depth+1, max_depth)
            future7 = pool.submit(_strassen, matrix_sub(A12, A22), matrix_add(B21, B22), threshold, depth+1, max_depth)

            M1 = future1.result()
            M2 = future2.result()
            M3 = future3.result()
            M4 = future4.result()
            M5 = future5.result()
            M6 = future6.result()
            M7 = future7.result()
    else:
        # Если максимальная глубина параллелизации достигнута, выполняем рекурсивно последовательно
        M1 = _strassen(matrix_add(A11, A22), matrix_add(B11, B22), threshold, depth+1, max_depth)
        M2 = _strassen(matrix_add(A21, A22), B11, threshold, depth+1, max_depth)
        M3 = _strassen(A11, matrix_sub(B12, B22), threshold, depth+1, max_depth)
        M4 = _strassen(A22, matrix_sub(B21, B11), threshold, depth+1, max_depth)
        M5 = _strassen(matrix_add(A11, A12), B22, threshold, depth+1, max_depth)
        M6 = _strassen(matrix_sub(A21, A11), matrix_add(B11, B12), threshold, depth+1, max_depth)
        M7 = _strassen(matrix_sub(A12, A22), matrix_add(B21, B22), threshold, depth+1, max_depth)

    # Комбинируем полученные результаты для формирования итоговых квадрантов
    C11 = matrix_add(matrix_sub(matrix_add(M1, M4), M5), M7)
    C12 = matrix_add(M3, M5)
    C21 = matrix_add(M2, M4)
    C22 = matrix_add(matrix_sub(matrix_add(M1, M3), M2), M6)

    return join_matrices(C11, C12, C21, C22)

def strassen_multiply(A, B, threshold=64, max_depth=2):
    """
    Обёртка для алгоритма Штрассена.
    
    Параметры:
      A, B      - входные квадратные матрицы (размер 2^n)
      threshold - порог для переключения на классическое умножение
      max_depth - максимальная глубина параллельной рекурсии
    """
    return _strassen(A, B, threshold, depth=0, max_depth=max_depth)

# Пример использования:
if __name__ == '__main__':
    # Пример: умножение 4x4 матриц (4 = 2^2)
    A = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]
    
    B = [
        [16, 15, 14, 13],
        [12, 11, 10, 9],
        [8, 7, 6, 5],
        [4, 3, 2, 1]
    ]
    
    C = strassen_multiply(A, B, threshold=2, max_depth=2)
    
    print("Результат умножения:")
    for row in C:
        print(row)
