import numpy as np
import pandas as pd

'''df = pd.read_csv('data.csv')  # CSV 파일 경로 지정
D = df.values.tolist()'''

# 수박 데이터세트
x1 = (0.697, 0.460)
x2 = (0.774, 0.376)
x3 = (0.634, 0.264)
x4 = (0.608, 0.318)
x5 = (0.556, 0.215)
x6 = (0.403, 0.237)
x7 = (0.481, 0.149)
x8 = (0.437, 0.211)
x9 = (0.666, 0.091)
x10 = (0.243, 0.267)
x11 = (0.245, 0.057)
x12 = (0.343, 0.099)
x13 = (0.639, 0.161)
x14 = (0.657, 0.198)
x15 = (0.360, 0.370)
x16 = (0.593, 0.042)
x17 = (0.719, 0.103)
x18 = (0.359, 0.188)
x19 = (0.339, 0.241)
x20 = (0.282, 0.257)
x21 = (0.748, 0.232)
x22 = (0.714, 0.346)
x23 = (0.483, 0.312)
x24 = (0.478, 0.437)
x25 = (0.525, 0.369)
x26 = (0.751, 0.489)
x27 = (0.532, 0.472)
x28 = (0.473, 0.376)
x29 = (0.725, 0.445)
x30 = (0.446, 0.459)
D = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30]

k = 5  # 클러스터 갯수 설정

m = len(D)
C = D.copy()
def distance(ci, cj):       # 거리 구하는 함수
    return np.linalg.norm(np.array(ci) - np.array(cj))


def find_min_distance(M):     # 최소 거리 인덱스 찾는 함수
    mask = np.tril(np.ones(M.shape), k=-1).astype(bool)  #행렬에서 하삼각 부분만 고려
    min_value = np.min(M[mask])             # 최솟값
    min_index = np.unravel_index(np.argmin(M[mask]), M.shape)     # 최솟값 인덱스 식별
    return min_value, min_index

def remove_row_and_column(M, i, j):    # 특정 행과 열을 삭제하는 함수
    M = np.delete(M, i, axis=0)    # i행 삭제
    M = np.delete(M, j, axis=1)    # j열 삭제
    return M                       # 행렬 M 반환

M = np.zeros((m, m))     # 행렬 M 초기화


for i in range(m):
    for j in range(m):
        M[i, j] = distance(C[i], C[j])      # 클래스별 거리를 구해서 M 행렬에 대입
        M[j, i] = M[i, j]
q = m

merge_history = []

# 최솟값 구해서 행렬 M 에서 삭제
while q > k:
    min_value, min_index = find_min_distance(M)
    i0 = min_index[0]
    j0 = min_index[1]
    C[i0] = np.mean([C[i0], C[j0]], axis=0)
    merge_history.append((i0, j0, min_value, q))
    C.pop(j0)

    M = remove_row_and_column(M, j0, j0)
    q -= 1

print("최종 클러스터:", C)
print("병합 이력:", merge_history)

import matplotlib.pyplot as plt


# 클러스터 결과 시각화
plt.figure(figsize=(10, 7))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
cluster_assignment = np.zeros(len(D))

for idx, point in enumerate(D):
    min_dist = float('inf')
    assigned_cluster = -1
    for cluster_idx, cluster_center in enumerate(C):
        dist = distance(point, cluster_center)
        if dist < min_dist:
            min_dist = dist
            assigned_cluster = cluster_idx
    cluster_assignment[idx] = assigned_cluster
    plt.scatter(point[0], point[1], c=colors[assigned_cluster % len(colors)])

plt.scatter([c[0] for c in C], [c[1] for c in C], c='black', marker='x', s=100)
plt.title('Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()




