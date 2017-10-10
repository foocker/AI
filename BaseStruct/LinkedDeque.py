class _DoublyLinkedBase:
    '''A base class providing a doubly linked list representation.'''

    class _Node:
        '''Lightweight, nonpublic class for storing a doubly linked node.'''
        __slots__ = '_element', '_prev', '_next'    # streamline memory

        def __init__(self, element, prev, nextt):
            self._element = element    # user's element
            self._prev = prev          # previous node reference
            self._next = nextt
        
    def __init__(self):
        '''Creat an empty list.'''
        self._header = self._Node(None, None, None)
        self._trailer = self._Node(None, None, None)
        self._header._next = self._trailer
        self._trailer._prev = self._header
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def _insert_between(self, e, predecessor, successor):
        '''Add element e between two exiting nodes and return new node.'''
        newest = self._Node(e, predecessor, successor)
        predecessor._next = newest
        successor._prev = newest
        self._size += 1
        return newest

    def _delete_node(self, node):
        '''Delete nonsentinel node from the list and return its element.'''
        predecessor = node._prev
        successor = node._next
        predecessor._next = successor
        successor._prev = predecessor
        self._size -= 1
        element = node._element
        node._prev = node._next = node._element = None    # deprecate node
        return element


class LinkedDeque(_DoublyLinkedBase):
    '''Double-ended queue implementation based on a doubly linked list.'''

    def first(self):
        '''Return(but do not remove)the element at the front of the deque.'''
        if self.is_empty():
            raise Exception('Deque is empty.')
        return self._header._next._element    # real item just after header

    def last(self):
        if self.is_empty():
            raise Exception('Deque is empty.')
        return self._trailer._prev._element

    def insert_first(self, e):
        '''Add an element to the front of the deque.'''
        self._insert_between(e, self._header, self._header._next)    # after header
    
    def insert_last(self, e):
        self._insert_between(e, self._trailer._prev, self._trailer)

    def delete_first(self):
        '''Remove and return the element from the front of the deque.'''
        if self.is_empty():
            raise Exception('Deque is empty.')
        return self._delete_node(self._header._next)

    def delete_last(self):
        '''Remove and return the element from the back of the deque.'''
        if self.is_empty():
            raise Exception('Deque is empty.')
        return self._delete_node(self._trailer._prev)

if __name__ == '__main__':
    LD = LinkedDeque()
    print(LD._header._element, LD._header._prev, LD._header._next._element)
    LD.insert_first(22),LD.insert_last(23),LD.insert_first('adc')
    print(LD._size)
    print(LD.first(), LD.last())
    LD.delete_first()
    print(LD.first())


    

        

        
