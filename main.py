from concurrent.futures import ProcessPoolExecutor
import random
def conventional_multiply(A, B):
 
    n = len(A)
    m = len(B[0])
    p = len(B)
    result = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i][j] += A[i][k] * B[k][j]
    return result

def add_matrix(A, B):
    
    n = len(A)
    m = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]

def sub_matrix(A, B):
    
    n = len(A)
    m = len(A[0])
    return [[A[i][j] - B[i][j] for j in range(m)] for i in range(n)]

def split_matrix(A):
  
    n = len(A)
    mid = n // 2
    A11 = [row[:mid] for row in A[:mid]]
    A12 = [row[mid:] for row in A[:mid]]
    A21 = [row[:mid] for row in A[mid:]]
    A22 = [row[mid:] for row in A[mid:]]
    return A11, A12, A21, A22

def join_quadrants(C11, C12, C21, C22):
   
    top = [r1 + r2 for r1, r2 in zip(C11, C12)]
    bottom = [r1 + r2 for r1, r2 in zip(C21, C22)]
    return top + bottom

def parallel_strassen(A, B, threshold=64):

    n = len(A)
    if n <= threshold:
        return conventional_multiply(A, B)
    

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)
  
    A11_plus_A22 = add_matrix(A11, A22)
    B11_plus_B22 = add_matrix(B11, B22)
    A21_plus_A22 = add_matrix(A21, A22)
    B12_minus_B22 = sub_matrix(B12, B22)
    B21_minus_B11 = sub_matrix(B21, B11)
    A11_plus_A12 = add_matrix(A11, A12)
    A21_minus_A11 = sub_matrix(A21, A11)
    B11_plus_B12 = add_matrix(B11, B12)
    A12_minus_A22 = sub_matrix(A12, A22)
    B21_plus_B22 = add_matrix(B21, B22)
    
   
    with ProcessPoolExecutor() as executor:
        f1 = executor.submit(parallel_strassen, A11_plus_A22, B11_plus_B22, threshold)
        f2 = executor.submit(parallel_strassen, A21_plus_A22, B11, threshold)
        f3 = executor.submit(parallel_strassen, A11, B12_minus_B22, threshold)
        f4 = executor.submit(parallel_strassen, A22, B21_minus_B11, threshold)
        f5 = executor.submit(parallel_strassen, A11_plus_A12, B22, threshold)
        f6 = executor.submit(parallel_strassen, A21_minus_A11, B11_plus_B12, threshold)
        f7 = executor.submit(parallel_strassen, A12_minus_A22, B21_plus_B22, threshold)
        
       
        M1 = f1.result()
        M2 = f2.result()
        M3 = f3.result()
        M4 = f4.result()
        M5 = f5.result()
        M6 = f6.result()
        M7 = f7.result()

    C11 = add_matrix(sub_matrix(add_matrix(M1, M4), M5), M7)
    C12 = add_matrix(M3, M5)
    C21 = add_matrix(M2, M4)
    C22 = add_matrix(sub_matrix(add_matrix(M1, M3), M2), M6)
    
    return join_quadrants(C11, C12, C21, C22)


def generate_square_matrix(n, min_value=0, max_value=100):
    return [[random.randint(min_value, max_value) for _ in range(n)] for _ in range(n)]




if __name__ == '__main__':
    n = 16
    threshold=2

    A = generate_square_matrix(n)
    B = generate_square_matrix(n)
    C = parallel_strassen(A, B, threshold=threshold)
    
    print("Результат алгоритма:")
    for row in C:
        print(row)
    
    C = conventional_multiply(A, B)
    print("Результат обычного умножения:")
    for row in C:
        print(row)
