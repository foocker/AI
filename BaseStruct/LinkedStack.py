class LinkedStack:
    ''' LIFO Stack implementation using singly linked list for storage'''

    class _Node:
        __slots__ = '_element', '_next'

        def __init__(self, element, next):
            self._element = element
            self._next = next
    
    def __init__(self):
        '''Creat an empty stack'''
        self._head = None
        self._size = 0
    
    def __len__(self):
        return self._size
    
    def is_empty(self):
        return self._size == 0

    def push(self, e):
        self._head = self._Node(e, self._head)
        self._size += 1
    
    def top(self):
        if self.is_empty():
            raise Exception('Stack is empty')
        return self._head._element

    def pop(self):
        '''Remove and return the element from the top of the stack.'''
        if self.is_empty():
            raise Exception('Stack is empty.')
        answer = self._head._element
        self._head = self._head._next
        self._size -= 1
        return answer

if __name__ == '__main__':
    linckstack_instance = LinkedStack()
    print(linckstack_instance._size)
    linckstack_instance.push(23)
    print(linckstack_instance.top())
    linckstack_instance.push(233)
    linckstack_instance.push('abc')
    print(linckstack_instance.pop())
    print(linckstack_instance.__len__())
    linckstack_instance.pop()
    linckstack_instance.pop()
    # linckstack_instance.pop()