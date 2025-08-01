from __future__ import annotations
import copy
from collections import deque
from collections import defaultdict
from typing import DefaultDict, List


"""
TODO:
- __init__ 구현하기
- add_edge 구현하기
- dfs 구현하기 (재귀 또는 스택 방식 선택)
- bfs 구현하기
"""


class Graph:
    def __init__(self, n: int) -> None:
        """
        그래프 초기화
        n: 정점의 개수 (1번부터 n번까지)
        """
        self.n = n
        # 구현하세요!
        self.graph: list[list[int]] = [[] for _ in range(n + 1)]

    
    def add_edge(self, u: int, v: int) -> None:
        """
        양방향 간선 추가
        u -> v, v-> u
        """
        # 구현하세요!
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def dfs(self, start: int) -> list[int]:
        """
        깊이 우선 탐색 (DFS)
        
        구현 방법 선택:
        1. 재귀 방식: 함수 내부에서 재귀 함수 정의하여 구현
        2. 스택 방식: 명시적 스택을 사용하여 반복문으로 구현
        -> 재귀 방식을 선택해 recur_search로 구현함
        
        - 방문한 노드 번호의 순서를 리스트로 반환
        """
        # 구현하세요!
        search = [False] * (self.n+1)
        result = []
        
        def recur_search(node: int):
            search[node] = True
            result.append(node)
            for neighbor in sorted(self.graph[node]):
                if not search[neighbor]:
                    recur_search(neighbor)
        
        recur_search(start)
        return result
    
    def bfs(self, start: int) -> list[int]:
        """
        너비 우선 탐색 (BFS)
        - 큐를 사용하여 구현
        - 시작 노드로부터 인접한 노드를 차례로 방문하여 탐색
        - 방문한 노드 번호의 순서를 리스트로 반환
        """
        # 구현하세요!
        search = [False] * (self.n+1)
        search[start] = True
        result = []
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in sorted(self.graph[node]):
                if not search[neighbor]:
                    search[neighbor] = True
                    queue.append(neighbor)
                    
        return result
        
    
    def search_and_print(self, start: int) -> None:
        """
        DFS와 BFS 결과를 출력
        """
        dfs_result = self.dfs(start)
        bfs_result = self.bfs(start)
        
        print(' '.join(map(str, dfs_result)))
        print(' '.join(map(str, bfs_result)))



from typing import Callable
import sys


"""
-아무것도 수정하지 마세요!
"""


def main() -> None:
    intify: Callable[[str], list[int]] = lambda l: [*map(int, l.split())]

    lines: list[str] = sys.stdin.readlines()

    N, M, V = intify(lines[0])
    
    graph = Graph(N)  # 그래프 생성
    
    for i in range(1, M + 1): # 간선 정보 입력
        u, v = intify(lines[i])
        graph.add_edge(u, v)
    
    graph.search_and_print(V) # DFS와 BFS 수행 및 출력


if __name__ == "__main__":
    main()
