class LinkedQueue:
    """FIFO queue implementation using a singly linked list for storage."""

    class _Node:
        __slots__ = '_element', '_next'

        def __init__(self, element, next):
            self._element = element
            self._next = next
    
    def __init__(self):
        '''Creat an empty queue.'''
        self._head = None
        self._tail = None
        self._size = 0
    
    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def first(self):
        if self.is_empty():
            raise Exception('Queue is empty')
        return self._head._element

    def dequeue(self):
        '''Remove and return the first element of the queue.(FIFO)'''
        if self.is_empty():
            raise Exception('Queue is empty')
        answer = self._head._element
        self._head = self._head._next
        self._size -= 1
        if self.is_empty():
            self._tail = None
        return answer

    def enqueue(self, e):
        '''Add an element to the back of the queue.'''
        newest = self._Node(e, None)
        if self.is_empty():
            self._head = newest
        else:
            self._tail._next = newest
        self._tail = newest    # update reference to tail node
        self._size += 1
    
if __name__ == '__main__':
    linkedqueue_instance = LinkedQueue()
    # print(linkedqueue_instance.first())
    linkedqueue_instance.enqueue('人')
    linkedqueue_instance.enqueue('狗')
    print(linkedqueue_instance.first())
    print(linkedqueue_instance.dequeue())


