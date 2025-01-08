"""
TokenTrie: hold the LLM token vocabulary in a prefix tree in otder to perform
operations over the whole vocabulary or parts of it in logarithmic time instead
of linear.
"""

from __future__ import annotations
from collections import namedtuple
from typing import Callable, Iterable, Tuple


TokenTrieStats = namedtuple(
    "TokenTrieStats", ["tokenids", "trienodes", "trieleaves", "triedepth"]
)


class TokenTrie:
    """
    Access the tokens in a vocabulary hierarchically by prefix.
    Ids are stored as a bitmap with bits set to one meaning id is present.
    """

    def __init__(self):
        self.children: dict[str, TokenTrie] = {}
        self.ids: int = 0

    def insert_all(self, vocabulary: Iterable[Tuple[int, str]]):
        """
        Insert all the tokens in the vocabulary in the trie, with the id of
        each token being its index in the vocabulary.
        """
        for _id, token in vocabulary:
            if len(token) > 0:
                self.insert(token, _id)

    def insert(self, token, _id):
        """
        Insert one token in the trie, with the given id.
        """
        if len(token) == 0:
            self.ids |= 1 << _id
        else:
            head, tail = token[0], token[1:]
            child = self.children.get(head, self.__class__())
            child.insert(tail, _id)
            self.children[head] = child

    def insert_ids(self, token, ids):
        """
        Insert a token in the trie, with the given id set.
        This is useful e.g. when collapsing multiple branches into one.
        """
        if len(token) == 0:
            self.ids |= ids
        else:
            head, tail = token[0], token[1:]
            child = self.children.get(head, self.__class__())
            child.insert_ids(tail, ids)
            self.children[head] = child

    def collect_ids(self) -> set[int]:
        """
        Returns a set with the ids of the token(s) in this node and all the
        nodes below it.
        """
        ids = self.ids
        for child in self.children.values():
            ids |= child.collect_ids()
        return ids

    def dfs(self, prefix="") -> Iterable[tuple[str, int]]:
        """
        Traverse the trie depth-first, yielding (token, ids) tuples.
        """
        if self.ids:
            yield (prefix, self.ids)
        for char, child in self.children.items():
            yield from child.dfs(prefix + char)

    def map(self, map_fn: Callable[[str, int], str]) -> TokenTrie:
        """
        Return a trie where the characters are mapped to other characters using a
        function. This is useful for example to collapse a tree into a smaller one
        by pruning or merging branches where the characters are equivalent for a
        particular use case. The mapping function is passed a character to map, and
        the recursion level in the tree, and it can return True to preserve the
        branch of the tree as is, None to prune it, or a replacement character.
        If the latter, the branch will be recursed upon and stored under the
        replacement branch.
        """
        return self._map(map_fn, self.__class__())

    def _map(
        self, map_fn: Callable[[str, int], str], mapped_trie: TokenTrie, level: int = 0
    ) -> TokenTrie:
        """
        Internal implementation of map()
        """
        mapped_trie.ids |= self.ids
        for char, child in self.children.items():
            mapped_char = map_fn(char, level)
            if mapped_char is True:
                # If the mapping function returns True, preserve the original branch
                mapped_trie.children[char] = child
            elif mapped_char is None:
                # If the mapping function returns None, prune the original branch
                pass
            else:
                # Map the branch to a new character, e.g. merge several chars into one
                mapped_child = mapped_trie.children.get(
                    mapped_char, mapped_trie.__class__()
                )
                # pylint: disable-next=protected-access
                mapped_trie.children[mapped_char] = child._map(
                    map_fn, mapped_child, level + 1
                )
        return mapped_trie

    def _id_count(self) -> int:
        """
        Returns the number of ids in this node
        """
        # FUTURE: self.ids.bit_count() available from Python 3.10 is said to be 6x faster
        return bin(self.ids).count("1")

    def max_depth(self) -> int:
        """
        Return the max depth of any branch on the trie, i.e. the length of the longest token.
        """
        return max((child.max_depth() for child in self.children.values()), default=0) + 1

    def stats(self) -> TokenTrieStats:
        """
        Compute and return statistics on the trie, for debugging purposes.
        """
        ids = self._id_count()
        nodes = 1
        leaves = 0
        depth = 0
        if len(self.children) == 0:
            leaves = 1
        else:
            for branch in self.children.values():
                branch_ids, branch_nodes, branch_leaves, branch_depth = branch.stats()
                ids += branch_ids
                nodes += branch_nodes
                leaves += branch_leaves
                depth = max(depth, branch_depth)
        return TokenTrieStats(
            tokenids=ids, trienodes=nodes, trieleaves=leaves, triedepth=depth + 1
        )

    def __repr__(self):
        id_count = self._id_count()
        child_count = len(self.children)
        return f"{super().__repr__()}({id_count=}, {child_count=})"
