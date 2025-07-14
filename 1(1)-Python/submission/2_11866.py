from __future__ import annotations
from collections import deque


"""
TODO:
- rotate_and_remove 구현하기 
"""


def create_circular_queue(n: int) -> deque[int]:
    """
    1부터 n까지의 숫자로 deque를 생성합니다.
    
    Args:
        n (int): 생성할 큐의 크기
    Returns:
        deque[int]: 1부터 n까지의 숫자를 가진 deque
    """
    return deque(range(1, n + 1))

def rotate_and_remove(queue: deque[int], k: int) -> int:
    """
    큐에서 k번째 원소를 제거하고 반환합니다.
    큐를 왼쪽으로 k-1번 회전한 후, 맨 앞의 원소를 제거하고 반환합니다.
    
    Args:
        queue (deque[int]): 원소가 담긴 deque
        k (int): 제거할 원소의 위치
    Returns:
        int: 제거된 원소
    """
    # 구현하세요!
    queue.rotate(-k+1)
    return queue.popleft() 




"""
TODO:
- josephus_problem 구현하기
    # 요세푸스 문제 구현
        # 1. 큐 생성
        # 2. 큐가 빌 때까지 반복
        # 3. 제거 순서 리스트 반환
"""


def josephus_problem(n: int, k: int) -> list[int]:
    """
    요세푸스 문제 해결
    n명 중 k번째마다 제거하는 순서를 반환
    
    [메소드 흐름]
    1. create_circular_queue를 활용해 queue를 생성
    2. 제거 순서 리스트를 위한 빈 리스트 생성
    3. rotate_and_remove를 활용해 queue가 빌 때까지 k번째 원소를 제거
    4. 제거한 원소를 제거 순서 리스트에 추가
    5. 제거 순서 리스트 반환
    
    Args:
        n (int): 사람의 수
        k (int): 제거할 사람의 순서
    
    Returns:
        list[int]: 제거된 사람의 번호가 순서대로 담긴 리스트
    """
    # 구현하세요!
    queue = create_circular_queue(n)
    removed = []
    while len(queue) > 0:
        out = rotate_and_remove(queue, k)
        removed.append(out)
    return removed

def solve_josephus() -> None:
    """입, 출력 format"""
    n: int
    k: int
    n, k = map(int, input().split())
    result: list[int] = josephus_problem(n, k)
    
    # 출력 형식: <3, 6, 2, 7, 5, 1, 4>
    print("<" + ", ".join(map(str, result)) + ">")

if __name__ == "__main__":
    solve_josephus()