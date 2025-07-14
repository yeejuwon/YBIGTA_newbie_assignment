from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
: 행렬의 특정 위치에 모듈러 연산을 활용해 값을 설정
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
: __matmul__을 사용하여 행렬의 거듭제곱을 계산
- __repr__ 구현하기
: 출력 형식에 맞춰 행렬을 문자열로 표현

"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        # 구현하세요!
        self.matrix[key[0]][key[1]] = value % self.MOD

    def __matmul__(self, matrix: Matrix) -> Matrix:
        """
        행렬의 곱셈을 구하는 함수
        
        Args:
            matrix (Matrix): 곱셈을 진행할 행렬
        Returns:
            Matrix: 곱셈 결과 행렬
        """
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        행렬의 거듭제곱을 구하는 함수
        
        Args:
            n (int): 거듭제곱의 지수
        Returns:
            Matrix: 거듭제곱 결과 행렬
        """
        # 구현하세요!
        result = Matrix.eye(self.shape[0])
        base = self.clone()
        
        while n > 0:
            if n % 2 == 1:
                result = result.__matmul__(base)
            base = base.__matmul__(base)
            n //= 2
        
        return result

    def __repr__(self) -> str:
        """
        행렬을 문자열로 표현하는 함수
        
        Args:
            self (Matrix): 행렬
        Returns:
            str: 행렬을 표현한 문자열
        """
        # 구현하세요!
        return '\n'.join([' '.join(map(str, row)) for row in self.matrix]) + '\n'


from typing import Callable
import sys


"""
-아무것도 수정하지 마세요!
"""


def main() -> None:
    intify: Callable[[str], list[int]] = lambda l: [*map(int, l.split())]

    lines: list[str] = sys.stdin.readlines()

    N, B = intify(lines[0])
    matrix: list[list[int]] = [*map(intify, lines[1:])]

    Matrix.MOD = 1000
    modmat = Matrix(matrix)

    print(modmat ** B)


if __name__ == "__main__":
    main()