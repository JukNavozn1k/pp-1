import multiprocessing

# Функции для операций с матрицами
def add_matrix(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def subtract_matrix(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

# Стандартное умножение матриц (для базовых случаев)
def multiply_matrix(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(A))) for j in range(len(B[0]))] for i in range(len(A))]

# Алгоритм Штрассена
def strassen(A, B):
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    mid = n // 2

    # Разделяем матрицы на 4 подматрицы
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    
    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]
    
    # Рекурсивно вычисляем 7 произведений
    P1 = strassen(A11, subtract_matrix(B12, B22))
    P2 = strassen(add_matrix(A11, A12), B22)
    P3 = strassen(add_matrix(A21, A22), B11)
    P4 = strassen(A22, subtract_matrix(B21, B11))
    P5 = strassen(add_matrix(A11, A22), add_matrix(B11, B22))
    P6 = strassen(subtract_matrix(A12, A22), add_matrix(B21, B22))
    P7 = strassen(subtract_matrix(A11, A21), add_matrix(B11, B12))

    # Собираем результирующую матрицу
    C11 = add_matrix(subtract_matrix(add_matrix(P5, P4), P2), P6)
    C12 = add_matrix(P1, P2)
    C21 = add_matrix(P3, P4)
    C22 = subtract_matrix(subtract_matrix(add_matrix(P5, P1), P3), P7)

    # Объединяем все части
    C = [C11[i] + C12[i] for i in range(len(C11))]
    C += [C21[i] + C22[i] for i in range(len(C21))]
    return C

# Параллельная версия с использованием multiprocessing
def parallel_strassen(A, B):
    n = len(A)
    if n <= 2:  # Для малых матриц используем стандартное умножение
        return multiply_matrix(A, B)

    mid = n // 2

    # Разделяем матрицы на 4 подматрицы
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    
    B11 = [row[:mid] for row in B[:mid]]
    B12 = [row[mid:] for row in B[:mid]]
    B21 = [row[:mid] for row in B[mid:]]
    B22 = [row[mid:] for row in B[mid:]]

    # Используем multiprocessing для параллельных вычислений
    with multiprocessing.Pool(7) as pool:
        results = pool.starmap(strassen, [
            (A11, subtract_matrix(B12, B22)),
            (add_matrix(A11, A12), B22),
            (add_matrix(A21, A22), B11),
            (A22, subtract_matrix(B21, B11)),
            (add_matrix(A11, A22), add_matrix(B11, B22)),
            (subtract_matrix(A12, A22), add_matrix(B21, B22)),
            (subtract_matrix(A11, A21), add_matrix(B11, B12))
        ])

        P1, P2, P3, P4, P5, P6, P7 = results

    # Собираем итоговую матрицу
    C11 = add_matrix(subtract_matrix(add_matrix(P5, P4), P2), P6)
    C12 = add_matrix(P1, P2)
    C21 = add_matrix(P3, P4)
    C22 = subtract_matrix(subtract_matrix(add_matrix(P5, P1), P3), P7)

    C = [C11[i] + C12[i] for i in range(len(C11))]
    C += [C21[i] + C22[i] for i in range(len(C21))]
    return C

if __name__ == '__main__':
    # Пример использования
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
    
    result = parallel_strassen(A, B)
    for row in result:
        print(row)
