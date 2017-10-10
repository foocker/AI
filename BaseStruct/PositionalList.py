class _DoublyLinkedBase:
    """A base class providing a doubly linked list representation"""

    class _Node:
        """Lightweight, nonpublic class for storing a doubly linked node."""
        __slots__ = '_element', '_prev', '_next'  # streamline memory

        def __init__(self, element, prev, nextt):
            self._element = element
            self._prev = prev
            self._next = nextt
    
    def __init__(self):
        '''Creat an empty list'''
        self._header = self._Node(None, None, None)
        self._trailer = self._Node(None, None, None)
        self._header._next = self._trailer  # trailer is after header
        self._trailer._prev = self._header  # header is before trailer
        self._size = 0

    def __len__(self):
        return self._size
    
    def is_empty(self):
        return self._size == 0

    def _insert_between(self, e, predecessor, successor):
        """Add element e between two existing nodes and return new node."""
        newest = self._Node(e, predecessor, successor)  # linked to neighbors
        predecessor._next = newest
        successor._prev = newest
        self._size += 1
        return newest

    def _delete_node(self, node):
        """ Delete nonsentinel node from the list and return its element."""
        predecessor = node._prev
        successor = node._next
        predecessor._next = successor
        successor._prev = predecessor
        self._size -= 1
        element = node._element
        node._prev = node._next = node._element = None  # deprecate node
        return element

class PositionalList(_DoublyLinkedBase):
    """A sequential container of elements alowing positional access."""
    class Position:
        """ An abstraction representing the location of single element."""

        def __init__(self, container, node):
            '''Constructor should not be invoked by user.'''
            self._container = container
            self._node = node

        def element(self):
            return self._node._element

        def __eq__(self, other):
            '''Return Ture if other is Position representing the same location.'''
            return type(other) is type(self) and other._node is self._node

        def __ne__(self, other):
            return not(self == other)

        #------utility method------
    def _validate(self, p):
        '''Return Position's node,or raise appropriate error if invalid.'''
        if not isinstance(p, self.Position):
            raise TypeError('p must be proper Position type')
        if p._container is not self:
            raise ValueError('p does not belong to this container')
        if p._node._next is None:
            raise ValueError('p is no longer valid')
        return p._node

    def _make_position(self, node):
        '''Return Position instance for given node'''
        if node is self._header or self._trailer:
            return None
        else:
            return self.Position(self, node)  # legitimate position ?
    
    #-----accessor-----
    def first(self):
        '''Return the first Position in the list'''
        return self._make_position(self._header._next)

    def last(self):
        return self._make_position(self._trailer._prev)

    def before(self, p):
        '''Return the Position just before Position p (or None if p is first).'''
        node = self._validate(p)
        return self._make_position(node._prev)

    def after(self, p):
        node = self._validate(p)
        return self._make_position(node._next)

    def __iter__(self):
        '''Generate a forward iteration of elements of the list.'''
        cursor = self.first()
        while cursor is not None:
            yield cursor.element()
            cursor = self.after(cursor)
    
    #-----mutators-----
    #override inherited version to return Position, rather than Node
    def _insert_between(self, e, predecessor, successor):
        '''Add element between existing nodes and return new Positon.'''
        node = super()._insert_between(e, predecessor, successor)
        return self._make_position(node)

    def add_first(self, e):
        '''Insert element e at the front of the list and return new Position.'''
        return self._insert_between(e, self._header, self._header._next)

    def add_last(self, e):
        return self._insert_between(e, self._trailer._prev, self._trailer)

    def add_before(self, p, e):
        """Insert element e into list before Position p and return new Position."""
        orginal = self._validate(p)
        return self._insert_between(e, orginal._prev, orginal)

    def add_after(self, p, e):
        orginal = self._validate(p)
        return self._insert_between(e, orginal, orginal._next)

    def delete(self, p):
        '''Remove and return the element at Position p.'''
        orginal = self._validate(p)
        return self._delete_node(orginal)

    def replace(self, p, e):
        '''Replace the element at Position p with e...'''
        orginal = self._validate(p)
        old_value = orginal._element
        orginal._element = e
        return old_value

if __name__ == '__main__':
    L = PositionalList()
    L.add_first(10)
    L.add_last(9)
    print(L.first(), L._header._element)
    print(L._size, L._header._element)
    













