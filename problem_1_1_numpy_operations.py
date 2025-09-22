"""
문제 1.1: NumPy를 활용한 벡터 및 행렬 연산
"""
import numpy as np

def problem_1_1():
    print("=== 문제 1.1: NumPy를 활용한 벡터 및 행렬 연산 ===")
    
    # 1. 벡터 생성
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print("-" * 30)

    # 2. 요소별 곱셈
    element_wise_product = v1 * v2
    print(f"요소별 곱셈 결과: {element_wise_product}")
    print("각 위치의 요소끼리 곱셈을 수행합니다. 결과는 같은 크기의 벡터입니다.")
    print(f"[1*4, 2*5, 3*6] = {element_wise_product}")
    print("-" * 30)

    # 3. 벡터 내적
    dot_product = np.dot(v1, v2)
    # dot_product_alt = v1 @ v2  # @ 연산자로도 내적/행렬곱 수행 가능
    print(f"벡터 내적 결과: {dot_product}")
    print("요소별 곱셈의 합입니다. 결과는 스칼라(하나의 숫자)입니다.")
    print(f"1*4 + 2*5 + 3*6 = 4 + 10 + 18 = {dot_product}")
    print("-" * 30)

    # 4. 행렬 생성
    m1 = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 행렬
    m2 = np.array([[7, 8], [9, 10], [11, 12]])  # 3x2 행렬
    print(f"m1 (shape: {m1.shape}):\n{m1}")
    print(f"m2 (shape: {m2.shape}):\n{m2}")
    print("-" * 30)

    # 5. 행렬 곱셈
    matrix_product = np.dot(m1, m2)
    # matrix_product_alt = m1 @ m2
    print(f"행렬 곱셈 결과 (shape: {matrix_product.shape}):\n{matrix_product}")
    
    # 행렬 곱셈 설명
    print("\n행렬 곱셈 계산 과정:")
    print("m1[0,0]*m2[0,0] + m1[0,1]*m2[1,0] + m1[0,2]*m2[2,0] =", 
          f"{m1[0,0]}*{m2[0,0]} + {m1[0,1]}*{m2[1,0]} + {m1[0,2]}*{m2[2,0]} = {matrix_product[0,0]}")
    print("m1[0,0]*m2[0,1] + m1[0,1]*m2[1,1] + m1[0,2]*m2[2,1] =", 
          f"{m1[0,0]}*{m2[0,1]} + {m1[0,1]}*{m2[1,1]} + {m1[0,2]}*{m2[2,1]} = {matrix_product[0,1]}")
    
    # 추가 연산 예제
    print("\n=== 추가 벡터 연산 예제 ===")
    
    # 벡터의 크기(길이) 계산
    v1_magnitude = np.linalg.norm(v1)
    v2_magnitude = np.linalg.norm(v2)
    print(f"v1의 크기: {v1_magnitude:.4f}")
    print(f"v2의 크기: {v2_magnitude:.4f}")
    
    # 벡터 정규화 (단위벡터 만들기)
    v1_normalized = v1 / v1_magnitude
    v2_normalized = v2 / v2_magnitude
    print(f"v1 정규화: {v1_normalized}")
    print(f"v2 정규화: {v2_normalized}")
    
    # 코사인 유사도 계산
    cosine_similarity = dot_product / (v1_magnitude * v2_magnitude)
    print(f"코사인 유사도: {cosine_similarity:.4f}")
    
    return {
        'vectors': (v1, v2),
        'element_wise_product': element_wise_product,
        'dot_product': dot_product,
        'matrices': (m1, m2),
        'matrix_product': matrix_product,
        'magnitudes': (v1_magnitude, v2_magnitude),
        'cosine_similarity': cosine_similarity
    }

if __name__ == "__main__":
    results = problem_1_1()