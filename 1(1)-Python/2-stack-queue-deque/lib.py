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