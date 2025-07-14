from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable


"""
TODO:
- Trie.push 구현하기
- (필요할 경우) Trie에 추가 method 구현하기
"""


T = TypeVar("T")


@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False


class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T의 열 (list[int]일 수도 있고 str일 수도 있고 등등...)

        action: trie에 seq을 저장하기
        """
        # 구현하세요!
        node_index = 0
        for item in seq:
            node = self[node_index]
            found = False
            for child_index in node.children:
                if self[child_index].body == item:
                    node_index = child_index
                    found = True
                    break
            if not found:
                new_node = TrieNode(body=item)
                self.append(new_node)
                new_index = len(self) - 1
                node.children.append(new_index)
                node_index = new_index
        self[node_index].is_end = True
    
    # 구현하세요!
    def count_subtries(self, index: int) -> int:
        """
        해당 노드에서 끝나는 단어의 수 + 자식 서브트리 내의 단어 수
        index: trie의 노드 인덱스
        """
        
        node = self[index]
        count = 1 if node.is_end else 0
        for child_index in node.children:
            count += self.count_subtries(child_index)
        return count
    
    def factorial(self, n: int) -> int:
        """
        n!을 계산하는 함수
        """
        result = 1
        for i in range(2, n+1):
            result *= i
        return result
    
    def count_permut(self, index: int) -> int:
        """
        문제에 주어진 규칙을 만족하는 수를 반환
        : 순열 계산
        """
        node = self[index]
        total = 0
        result = 1
        MOD = 1000000007
        
        for child_index in node.children:
            sub_count = self.count_subtries(child_index)
            sub_result = self.count_permut(child_index)
            total += sub_count
            result = (result * sub_result) % MOD
        
        if total > 0:
            result = (result * self.factorial(total)) % MOD
        
        return result


import sys


"""
TODO:
- 일단 lib.py의 Trie Class부터 구현하기
- main 구현하기

힌트: 한 글자짜리 자료에도 그냥 str을 쓰기에는 메모리가 아깝다...
"""


def main() -> None:
    # 구현하세요!
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    names = data[1:n+1]
    
    trie = Trie[int]()
    for name in sorted(names):
        trie.push([ord(c) for c in name])

    print(trie.count_permut(0))
    


if __name__ == "__main__":
    main()