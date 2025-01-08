"""
Base token acceptors.

A token acceptor constrains the tokens that are acceptable at this point in
the parsing or generation of a text.

Since multiple parses of a given input may be possible (or multiple generations
valid according to e.g. a schema), the acceptor creates multiple "cursors", one
for each valid current state of the acceptor. This is akin to a chart parser,
where all possible parses of the input are carried in parallel, which minimizes
backtracking that is expensive on an LLM.

The basic flow is:
- First, the vocabulary (list of possible tokens for the LLM) is prepared into
  a trie for logarithmic traversal. Subclasses may also perform their own
  vocabulary preparation.
- The acceptor's get_cursors() method is called, and the acceptor issues one or
  more cursors with initial state(s).
- The trie is traversed to find which tokens are a valid match in the current
  state of the active cursor(s). For each cursor:
  - The select() method is called to narrow down the next character(s) that the
    cursor can accept in its current state.
  - For each selected character, we advance() the cursor, obtaining one or more
    follow-up cursors that represent the next state(s) of the cursor.
  - We descend down the trie branch corresponding to the selected character, and
    perform the same select(), advance() operation on the new cursor(s).
  - We traverse until the cursor(s) have reached an accepted state or we reach a
    leaf node.
  - As we traverse the trie recursively, we collect the token ids for each node.
    This creates a set of valid tokens that will be accepted.

For example: if we have a TextAcceptor that will accept the word "true", the
initial cursor's select() method will return "t" as the set of acceptable
characters. We will then advance the cursor and obtain a cursor that accepts the
word "rue", and our current trie node will become the "t" child branch of the
prior trie node. We will then match the new trie node with the new acceptor, etc.

Acceptors can be chained with e.g. a StateMachineAcceptor. In this case, when a
cursor reaches a final state, the parent acceptor moves its own cursor forward,
potentially issuing more cursors that can be matched with the remainder of the
trie.

Some methods have been added to help prevent combinatorial explosions while
searching that can have a big effect in performance. For example, an acceptor
for a quoted string can select() a very large amount of characters after the
first quote. Descending upon every branch of the trie is not necessary in as
much as every character is essentially equivalently valid. To avoid this, we
allow the acceptor to prune the trie so that all equivalent characters are
collapsed into one branch. In such a collapsed trie, each node keeps a set with
all the ids for valid tokens of the same length, which are equivalent from the
point of the view of the acceptor.
"""

from __future__ import annotations
from copy import copy as shallowcopy
from time import time_ns
from typing import Iterable, Tuple

from .util.tokentrie import TokenTrie


class TokenAcceptor:
    """
    Base class for token acceptors.
    """

    @classmethod
    def prepare_vocabulary(cls, vocabulary: Iterable[Tuple[int, str]]) -> TokenTrie:
        """
        Given a list of tokens (typically the vocabulary of an LLM), create
        a trie that will be used to select the tokens accepted by the current
        set of cursors.
        """
        vocabulary_trie = TokenTrie()
        vocabulary_trie.insert_all(vocabulary)
        return vocabulary_trie

    @classmethod
    def match_all(cls, cursors: Iterable[TokenAcceptor.Cursor], trie: TokenTrie) -> int:
        """
        Find which tokens in the vocabulary move any of the cursors towards an
        acceptance state from their current state.
        """
        if any(cursor.matches_all() for cursor in cursors):
            return trie.collect_ids()
        bitmap = 0
        for cursor in cursors:
            bitmap |= cursor.match(trie)
        return bitmap

    @classmethod
    def debug_match_all(
        cls,
        cursors: Iterable[TokenAcceptor.Cursor],
        trie: TokenTrie,
        debug_output_fn=print,
    ) -> int:
        """
        Same as match_all() but outputs debug information.
        """
        if any(cursor.matches_all() for cursor in cursors):
            return trie.collect_ids()
        debug_output_fn("MATCH ALL")
        bitmap = 0
        for cursor in cursors:
            start = time_ns()
            cursor_matches = cursor.debug_match(trie, debug_output_fn)
            dt_ns = time_ns() - start
            match_count = bin(cursor_matches).count("1")
            debug_output_fn(f"t={dt_ns/1e6:.02f}ms {match_count=} {repr(cursor)}")
            bitmap |= cursor_matches
        return bitmap

    @classmethod
    def advance_all(
        cls, cursors: Iterable[TokenAcceptor.Cursor], char: str
    ) -> list[TokenAcceptor.Cursor]:
        """
        Advance multiple cursors in parallel.
        """
        return [
            new_cursor
            for cursor in cursors
            if char in cursor.select(set(char))
            for new_cursor in cursor.advance(char)
        ]

    def get_cursors(self) -> Iterable[TokenAcceptor.Cursor]:
        """
        Get one or more cursors to traverse the acceptor.
        Override.
        """
        return [self.__class__.Cursor(self)]

    class Cursor:
        """
        A cursor encapsulates a valid current state of a token acceptor.
        """

        def __init__(self, acceptor: TokenAcceptor):
            pass

        def clone(self):
            """
            Cursors are never mutated, they are cloned as they advance.
            They should also be lightweight: think twice before overriding this
            to e.g. a deepcopy.
            """
            return shallowcopy(self)

        def matches_all(self) -> bool:
            """
            The acceptor accepts all the tokens (i.e. free text).
            This is an optimization and only useful for acceptors that don't constrain
            the input, such as WaitForAcceptor.
            """
            return False

        def select(self, candidate_chars: set[str]) -> Iterable[str]:
            """
            Narrow down the characters that are offered to the cursor for advancement.
            This is a crucial performance improvement for cursors in a state where they'll
            accept only a small set of characters, since they will be tested against that
            set instead of the whole range of characters available.
            Override.
            """
            return candidate_chars

        # pylint: disable-next=unused-argument
        def advance(self, char: str) -> Iterable[TokenAcceptor.Cursor]:
            """
            If the character can be consumed, return new cursor(s) for the possible
            continuation(s). IMPORTANT: Cursors should not mutate their state, only
            return mutated copies of the object, as the advance method is called
            multiple times with different inputs. See clone() method above.
            Override.
            """
            return []

        def in_accepted_state(self) -> bool:
            """
            Returns True if the cursor has reached a final state.
            Typically, rather than override you should return an AcceptedState object
            in the advance() method when the state is reached after consuming input.
            """
            return False

        def get_value(self):
            """
            Returns the current value of the cursor as defined by itself. This can be
            either the ongoing representation of its temporary state, or its final value
            usable for the application once it reaches accepted state. At that point,
            cursors that return the same value are considered identical and duplicates
            may be discarded for performance.
            Override.
            """
            return None

        def get_value_path(self):
            """
            Returns the path of the value being pointed at by the cursor as defined by the
            application. This can be for example a JSON path in the case of a JSON acceptor.
            For higher-level application purposes only, not required for accepting.
            Override.
            """
            return ""

        def is_in_value(self):
            """
            Returns true if the cursor is accepting a value as opposed to syntactic elements.
            Used in conjunction with get_value_path().
            Override.
            """
            return False

        def prune(self, trie: TokenTrie) -> Iterable[(str, TokenTrie)]:
            """
            Select the children of the trie to search for matches. See match() below.
            This can be overriden in order to e.g. use a collapsed trie.
            """
            if trie.children:
                chars = set(trie.children.keys())
                selected_chars = chars & set(self.select(chars))
                for char in selected_chars:
                    yield (char, trie.children[char])

        def match(self, trie: TokenTrie) -> int:
            """
            Find which tokens in the vocabulary move the acceptor towards an acceptance
            state from the current state held by this cursor.
            Returns a bit map with the bits corresponding to the index if the matched
            tokens set to 1.
            """
            if self.matches_all():
                return trie.collect_ids()
            bitmap = 0
            for char, child in self.prune(trie):
                followup_cursors = self.advance(char)
                if followup_cursors:
                    bitmap |= child.ids
                    for followup_cursor in followup_cursors:
                        bitmap |= followup_cursor.match(child)
            return bitmap

        def debug_match(
            self, trie: TokenTrie, debug_output_fn=print, debug_indent=1
        ) -> int:
            """
            Same as match() but outputs debug information
            """
            debug_start = time_ns()
            if self.matches_all():
                return trie.collect_ids()
            bitmap = 0
            debug_label = type(self).__qualname__
            if isinstance(self, StateMachineAcceptor.Cursor):
                debug_label += f"({type(self.transition_cursor).__qualname__})"
            debug_prefix = "  " * debug_indent + debug_label
            debug_prune_start = time_ns()
            for char, child in self.prune(trie):
                debug_advance_start = time_ns()
                followup_cursors = self.advance(char)
                debug_advance_end = time_ns()
                prune_time = (debug_advance_start - debug_prune_start) / 1e6
                advance_time = (debug_advance_end - debug_advance_start) / 1e6
                debug_output_fn(
                    f"{debug_prefix} >>> "
                    f"{prune_time=:.02f}ms {advance_time=:.02f}ms char={repr(char)}"
                )
                debug_followup_start = time_ns()
                if followup_cursors:
                    bitmap |= child.ids
                    for followup_cursor in followup_cursors:
                        bitmap |= followup_cursor.debug_match(
                            child, debug_output_fn, debug_indent + 1
                        )
                debug_followup_end = time_ns()
                followup_time = (debug_followup_end - debug_followup_start) / 1e6
                followup_count = len(followup_cursors)
                match_count = bin(bitmap).count("1")
                debug_output_fn(
                    f"{debug_prefix} <<< {followup_count=} {followup_time=:.02f}ms {match_count=}"
                )
                debug_prune_start = time_ns()
            total_time = (time_ns() - debug_start) / 1e6
            debug_output_fn(f"{debug_prefix} {total_time=:.02f}ms")
            return bitmap

        def __repr__(self):
            return f"{type(self).__qualname__}(value={repr(self.get_value())})"


class AcceptedState(TokenAcceptor.Cursor):
    """
    Holds a cursor that has reached the accepted state.
    """

    def __init__(self, cursor: TokenAcceptor.Cursor):
        self.cursor = cursor

    def in_accepted_state(self):
        return True

    def get_value(self):
        return self.cursor.get_value()

    def __repr__(self):
        return f"✅{repr(self.cursor)}"


class CharAcceptor(TokenAcceptor):
    """
    Accept one character iff is in the set of expected characters.
    """

    def __init__(self, charset: Iterable[str]):
        self.charset = charset

    class Cursor(TokenAcceptor.Cursor):
        """
        Cursor for CharAcceptor
        """

        def __init__(self, acceptor, value=None):
            self.acceptor = acceptor
            self.value = value

        def select(self, candidate_chars):
            return self.acceptor.charset

        def advance(self, char):
            # Because we implemented the select method, we are guaranteed that the
            # char is in our accepted set.
            return [AcceptedState(self.__class__(self.acceptor, char))]

        def get_value(self):
            return self.value

        def __repr__(self):
            return f"charset={repr(self.acceptor.charset)} value={repr(self.value)}"


class TextAcceptor(TokenAcceptor):
    """
    Accept a pre-determined string of characters.
    """

    def __init__(self, text: str):
        assert len(text) > 0
        self.text = text

    class Cursor(TokenAcceptor.Cursor):
        """
        Cursor for TextAcceptor
        """

        def __init__(self, acceptor, pos=0):
            self.acceptor = acceptor
            self.pos = pos

        def select(self, candidate_chars):
            return self.acceptor.text[self.pos]

        def advance(self, char):
            next_cursor = self.__class__(self.acceptor, self.pos + 1)
            if next_cursor.pos == len(self.acceptor.text):
                return [AcceptedState(next_cursor)]
            return [next_cursor]

        def get_value(self) -> str:
            head = self.acceptor.text[0 : self.pos]
            tail = self.acceptor.text[self.pos :]
            if len(tail):
                return f"{head}👉{tail}"
            else:
                return f"{head}"


class StateMachineAcceptor(TokenAcceptor):
    """
    Token acceptor that follows a state graph that defines edges to transition
    from state to state. Each state can have multiple edges, defined by the
    target state and a TokenAcceptor that, when reaching accepted state, causes
    the state machine acceptor to move to the target state. This is repeated
    until the state machine reaches a final state. Multiple transition paths
    are explored in parallel.
    """

    def __init__(self, graph=None, initial_state=None, end_states=None):
        self.graph = graph or []
        self.initial_state = initial_state or 0
        self.end_states = set(end_states or ["$"])

    def get_edges(self, state):
        """
        Retrieve the graph edges for transitions out of this state.
        Can be overriden for dynamic graphs.
        """
        return self.graph[state]

    def get_cursors(self):
        initial_cursor = self.Cursor(self)
        initial_cursor.current_state = self.initial_state
        return self._find_transitions(initial_cursor, [], set())

    def _find_transitions(self, cursor, visited_states, traversed_edges):
        try:
            edges = self.get_edges(cursor.current_state)
        except (KeyError, IndexError, TypeError):
            assert cursor.current_state in self.end_states
            return []
        cursors = []
        for transition_acceptor, target_state in edges:
            if cursor.start_transition(transition_acceptor, target_state):
                for transition_cursor in transition_acceptor.get_cursors():
                    copy = cursor.clone()
                    copy.transition_cursor = transition_cursor
                    copy.target_state = target_state
                    # Handle cursors that start in an accepted state,
                    # e.g. EmptyTransition, WhitespaceAcceptor
                    if transition_cursor.in_accepted_state():
                        new_visited_states = visited_states + [cursor.current_state]
                        assert target_state not in new_visited_states  # Infinite loop
                        cursors += self._cascade_transition(
                            copy, new_visited_states, traversed_edges
                        )
                    else:
                        cursors.append(copy)
        return cursors

    def _cascade_transition(self, cursor, visited_states, traversed_edges):
        assert cursor.transition_cursor.in_accepted_state()
        # Copy before validation to allow for cursor mutation, e.g. storing the transition_value
        cursors = []
        copy: StateMachineAcceptor.Cursor = cursor.clone()
        if copy.complete_transition(
            copy.transition_cursor.get_value(),
            copy.target_state,
            copy.target_state in copy.acceptor.end_states,
        ):
            copy.current_state = copy.target_state
            copy.target_state = None
            copy.accept_history = copy.accept_history + [copy.transition_cursor.cursor]
            copy.transition_cursor = None
            copy.consumed_character_count = 0
            # De-duplicate cursors that have reached the same state with the same value.
            # This prevents combinatorial explosion because of e.g. empty transitions.
            state_value = (copy.current_state, repr(copy.get_value()))
            if state_value not in traversed_edges:
                traversed_edges.add(state_value)
                if copy.current_state in self.end_states:
                    cursors.append(AcceptedState(copy))
                cursors += self._find_transitions(copy, visited_states, traversed_edges)
        return cursors

    def advance_cursor(self, cursor, char):
        """
        Advance a cursor, and if it reaches accepted state, cause the state machine to transition.
        """
        next_cursors = []
        traversed_edges = set()
        for followup_cursor in cursor.transition_cursor.advance(char):
            copy = cursor.clone()
            copy.transition_cursor = followup_cursor
            copy.consumed_character_count += 1
            if followup_cursor.in_accepted_state():
                next_cursors += self._cascade_transition(
                    copy, [], traversed_edges
                )
            else:
                next_cursors.append(copy)
        return next_cursors

    class Cursor(TokenAcceptor.Cursor):
        """
        Cursor for StateMachineAcceptor
        """

        def __init__(self, acceptor):
            self.acceptor = acceptor
            self.accept_history = []
            self.current_state = None
            self.transition_cursor = None
            self.target_state = None
            self.consumed_character_count = 0

        def matches_all(self):
            if self.transition_cursor is None:
                return False
            return self.transition_cursor.matches_all()

        def select(self, candidate_chars):
            if self.transition_cursor is None:
                return set()
            return self.transition_cursor.select(candidate_chars)

        def prune(self, trie):
            if self.transition_cursor is None:
                return []
            return self.transition_cursor.prune(trie)

        def advance(self, char):
            return self.acceptor.advance_cursor(self, char)

        # pylint: disable-next=unused-argument
        def start_transition(self, transition_acceptor, target_state) -> bool:
            """
            Override to prevent an edge to be traversed.
            """
            return True

        def complete_transition(  # pylint: disable-next=unused-argument
            self, transition_value, target_state, is_end_state
        ) -> bool:
            """
            Override to perform additional checks on the acceptee and mutate the cursor
            with the transition_value as appropriate.
            """
            return True

        def get_value(self):
            value = [
                accepted_transition_cursor.get_value()
                for accepted_transition_cursor in self.accept_history
            ]
            if self.transition_cursor is not None:
                value.append(self.transition_cursor.get_value())
            return value

        def is_in_value(self):
            if self.consumed_character_count > 0:
                return self.transition_cursor.is_in_value()
            return self.accept_history[-1].is_in_value() if self.accept_history else None

        def get_value_path(self):
            if self.consumed_character_count > 0:
                return self.transition_cursor.get_value_path()
            return self.accept_history[-1].get_value_path() if self.accept_history else ""

        def __repr__(self) -> str:
            if self.transition_cursor is not None:
                transition_cursor = repr(self.transition_cursor)
                target_state = self.target_state
            else:
                transition_cursor = "None"
                target_state = "None"
            if self.accept_history:
                accept_history = []
                for accepted_transition_cursor in self.accept_history:
                    if isinstance(
                        accepted_transition_cursor, StateMachineAcceptor.Cursor
                    ):
                        accept_history += accepted_transition_cursor.accept_history
                    else:
                        accept_history.append(accepted_transition_cursor)
                history = repr(
                    "".join(
                        [
                            str(accepted_transition_cursor.get_value())
                            for accepted_transition_cursor in accept_history
                        ]
                    )
                )
            else:
                history = ""
            state = (
                f"{history} {self.current_state}⇒{target_state}  {transition_cursor}"
            )
            return f"{type(self).__qualname__}({state})"

    class EmptyTransitionAcceptor(TokenAcceptor):
        """
        Faux acceptor that allows to create empty transition edges in a state
        machine graph for convenience in expressing complex graphs.
        An empty edge skips the current state altogether, without the need to
        consume input.
        """

        def get_cursors(self):
            return [AcceptedState(self.Cursor(self))]

        class Cursor(TokenAcceptor.Cursor):
            """
            Cursor for EmptyTransitionAcceptor
            """

            def get_value(self):
                return ""

    # Singleton EmptyTransitionAcceptor
    EmptyTransition = EmptyTransitionAcceptor()


class SequenceAcceptor(StateMachineAcceptor):
    """
    Chain acceptors in sequence
    """

    def __init__(self, acceptors):
        graph = [[(acceptor, i + 1)] for i, acceptor in enumerate(acceptors)]
        super().__init__(graph, initial_state=0, end_states=[len(acceptors)])

    class Cursor(StateMachineAcceptor.Cursor):
        """
        Cursor for SequenceAcceptor. Defined for inspectability.
        """


class WaitForAcceptor(TokenAcceptor):
    """
    Accept all text until finding a segment that triggers another acceptor.
    This is useful to allow for free text until a delimiter is found, e.g.
    when the output of an LLM includes JSON that is encapsulated within a
    ```json ... ``` block.
    """

    def __init__(self, wait_for_acceptor: TokenAcceptor):
        self.wait_for_acceptor = wait_for_acceptor

    class Cursor(TokenAcceptor.Cursor):
        """
        Cursor for WaitForAcceptor
        """

        def __init__(self, acceptor, cursors=None):
            self.acceptor = acceptor
            if cursors:
                self.cursors = cursors
            else:
                self.cursors = acceptor.wait_for_acceptor.get_cursors()

        def matches_all(self):
            return True

        def advance(self, char):
            cursors = TokenAcceptor.advance_all(self.cursors, char)
            accepted_cursors = [
                cursor for cursor in cursors if cursor.in_accepted_state()
            ]
            if accepted_cursors:
                return accepted_cursors
            return [self.__class__(self.acceptor, cursors)]

        def get_value(self):
            return f"Waiting for {repr(self.cursors)}"
