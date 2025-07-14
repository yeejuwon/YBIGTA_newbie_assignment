from lib import Trie
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