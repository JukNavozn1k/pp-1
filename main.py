import random
import multiprocessing

def matrix_add(A, B):
    """Складывает две матрицы A и B."""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

def matrix_sub(A, B):
    """Вычитает матрицу B из матрицы A."""
    n = len(A)
    return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

def naive_mult(A, B):
    """Наивное умножение матриц (тройной цикл)."""
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def split_matrix(A):
    """Разбивает матрицу A на 4 подматрицы: A11, A12, A21, A22."""
    n = len(A)
    mid = n // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    return A11, A12, A21, A22

def combine_quadrants(A11, A12, A21, A22):
    """Объединяет 4 подматрицы в одну матрицу."""
    top = [a + b for a, b in zip(A11, A12)]
    bottom = [a + b for a, b in zip(A21, A22)]
    return top + bottom

def strassen(A, B, threshold=64, parallel=True):
    """
    Рекурсивное умножение матриц методом Штрассена.
    
    Параметры:
      A, B      - матрицы, представленные списками списков
      threshold - при размерах <= threshold используется наивное умножение
      parallel  - использовать параллелизм только на верхнем уровне рекурсии
    """
    n = len(A)
    if n <= threshold:
        return naive_mult(A, B)
    
    # Разбиваем матрицы на 4 подматрицы
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
    
    if parallel:
        # Параллельное вычисление 7 произведений методом multiprocessing
        pool = multiprocessing.Pool(processes=7)
        # Вызываем рекурсивно, передавая parallel=parallel для избежания порождения новых процессов
        async_M1 = pool.apply_async(strassen, (matrix_add(A11, A22), matrix_add(B11, B22), threshold, parallel))
        async_M2 = pool.apply_async(strassen, (matrix_add(A21, A22), B11, threshold, parallel))
        async_M3 = pool.apply_async(strassen, (A11, matrix_sub(B12, B22), threshold, parallel))
        async_M4 = pool.apply_async(strassen, (A22, matrix_sub(B21, B11), threshold, parallel))
        async_M5 = pool.apply_async(strassen, (matrix_add(A11, A12), B22, threshold, parallel))
        async_M6 = pool.apply_async(strassen, (matrix_sub(A21, A11), matrix_add(B11, B12), threshold, parallel))
        async_M7 = pool.apply_async(strassen, (matrix_sub(A12, A22), matrix_add(B21, B22), threshold, parallel))
        
        # Получаем результаты
        M1 = async_M1.get()
        M2 = async_M2.get()
        M3 = async_M3.get()
        M4 = async_M4.get()
        M5 = async_M5.get()
        M6 = async_M6.get()
        M7 = async_M7.get()
        pool.close()
        pool.join()
    else:
        M1 = strassen(matrix_add(A11, A22), matrix_add(B11, B22), threshold, parallel)
        M2 = strassen(matrix_add(A21, A22), B11, threshold, parallel)
        M3 = strassen(A11, matrix_sub(B12, B22), threshold, parallel)
        M4 = strassen(A22, matrix_sub(B21, B11), threshold, parallel)
        M5 = strassen(matrix_add(A11, A12), B22, threshold, parallel)
        M6 = strassen(matrix_sub(A21, A11), matrix_add(B11, B12), threshold, parallel)
        M7 = strassen(matrix_sub(A12, A22), matrix_add(B21, B22), threshold, parallel)
    
    # Вычисляем квадранты результирующей матрицы по формулам Штрассена:
    # C11 = M1 + M4 - M5 + M7
    C11 = matrix_add(matrix_sub(matrix_add(M1, M4), M5), M7)
    # C12 = M3 + M5
    C12 = matrix_add(M3, M5)
    # C21 = M2 + M4
    C21 = matrix_add(M2, M4)
    # C22 = M1 - M2 + M3 + M6
    C22 = matrix_add(matrix_add(matrix_sub(M1, M2), M3), M6)
    
    return combine_quadrants(C11, C12, C21, C22)

def generate_random_matrix(n, max_val=10):
    """Генерирует квадратную матрицу n×n со случайными целыми числами от 0 до max_val-1."""
    return [[random.randint(0, max_val - 1) for _ in range(n)] for _ in range(n)]

def print_matrix(M):
    """Выводит матрицу построчно."""
    for row in M:
        print(" ".join(map(str, row)))

if __name__ == '__main__':
    # Размер матрицы (для простоты должен быть степенью двойки, например, 4)
    n = 2
    A = generate_random_matrix(n)
    B = generate_random_matrix(n)
    
    print("Matrix A:")
    print_matrix(A)
    print("\nMatrix B:")
    print_matrix(B)
    
    # Умножение матриц с помощью алгоритма Штрассена с параллелизмом
    # Для демонстрации порог выставляем достаточно низким (например, 2)
    C = strassen(A, B, threshold=2, parallel=True)
    
    print("\nResult matrix C (A x B) by Strassen:")
    print_matrix(C)
    
    # Вычисляем результат стандартным наивным алгоритмом для проверки корректности
    C_ref = naive_mult(A, B)
    print("\nReference result (naive multiplication):")
    print_matrix(C_ref)
    
    print("\nResults match:", C == C_ref)
