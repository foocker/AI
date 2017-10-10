class Tree:
    '''Abstract base class representing a tree structure.'''
    # ------ nested Position class -----
    class Position:
        '''An abstraction representing the location of a single element.'''
        def element(self):
            '''Return the element stored at this Position.'''
            raise NotImplementedError('must be implemented by subclass')

        def __eq__(self, other):
            '''Return True is other Position represents the same location.'''
            raise NotImplementedError('must be implemented by subclass')
        
        def __ne__(self, other):
            '''Return True if other does not represent the same location.'''
            return not(self == other)    # opposite of __eq__

    # ---abstract methods that concrete subclass must support---
    def root(self):
        '''Return Position representing the tree's root(or None if p is root).'''
        raise NotImplementedError('must be implemented by subclass')

    def parent(self, p):
        '''Return Position representing p's parent(or None if p is root).'''
        raise NotImplementedError('must be implemented by subclass')
    
    def num_children(self, p):
        '''Return the number of children that Position p has.'''
        raise NotImplementedError('must be implemented by subclass')
    
    def children(self, p):
        '''Generate an iteration of Positions representing p's children.'''
        raise NotImplementedError('must be implemented by subclass')
    
    def __len__(self):
        '''Return the total number of elements in the tree.'''
        raise NotImplementedError('must be implemented by subclass')
    
    # --- concrete methods implemented in this class. ---
    def is_root(self, p):
        return self.root() == p

    def is_leaf(self, p):
        return self.num_children(p) == 0

    def is_empty(self, p):
        return len(self) == 0

    def depth(self, p):
        '''Return the number of levels separating Position p from the root.'''
        if self.is_root(p):
            return 0
        return 1 + self.depth(self.parent(p))

    def height(self, p=None):
        '''Return the height of subtree rooted at Position p. If p is None, return the height of the entire tree.'''
        return self._height2(p)

    def _height2(self, p):
        '''Return the height of the subtree rooted at Position p.'''
        if self.is_leaf(p):
            return 0
        else:
            return 1 + max(self._height2(c) for c in self.children(p))
    
class BinaryTree(Tree):
    '''Abstract base class representing a binary tree structure.'''
    # ----- additional abstract methods -----
    def left(self, p):
        '''Return a Position representing p's left child.  None'''
        raise NotImplementedError('must be implemented by subclass')
    
    def right(self, p):
        raise NotImplementedError('must be implemented by subclass')
    
    def sibling(self, p):
        '''Return a Position representing p's sibling (of None if no sibling).'''
        parent = self.parent(p)
        if parent is None:
            return None
        else:
            if p == self.left(parent):
                return self.right(parent)
            else:
                return self.left(parent)
    
    def children(self, p):
        '''Generate an iteration of Positions representing p's children.'''
        if self.left(p) is not None:
            yield self.left(p)
        if self.right(p) is not None:
            yield self.right(p)


class LinkedBinaryTree(BinaryTree):
    '''Linked representation of binary tree structure.'''

    class _Node:    # Lightweight, nonpublic class for storing a node
        __slots__ = '_element', '_parent', '_left', '_right'
        def __init__(self, element, parent=None, left=None, right=None):
            self._element = element
            self._parent = parent
            self._left = left
            self._right = right
    
    class Position(BinaryTree.Position):
        '''An abstraction representing the location of single element.'''
        def __init__(self, container, node):
            '''Constructor should not be invoked by user.'''
            self._container = container
            self._node = node
        
        def element(self):
            '''Return the element stored at this Position.'''
            return self._node._element

        def __eq__(self, other):
            '''Return True if other is a Position representating the same location.'''
            return type(other) == type(self) and other._node is self._node

    def _validate(self, p):
        '''Return associated node, if position is valid.'''
        if not isinstance(p, self.Position):
            raise TypeError('p must be proper Position type')
        if p._container is not self:
            raise ValueError('p does not belong to this container')
        if p._node._parent is p._node:
            raise ValueError('p is no longer valid')
        return p._node

    def _make_position(self, node):
        '''Return Position instance for given node(or None if no node).'''
        return self.Position(self, node) if node is not None else None

def __init__(self):
    '''Create an initially empty binary tree.'''
    self._root = None
    self._size = 0

    # ------ public accessors ------
    def __len__(self):
        '''Return thr total number of elements in the tree.'''
        return self._size

    def parent(self, p):
        '''Return the Position of p's parent(or None if p is root).'''
        node = self._validate(p)
        return self._make_position(node._parent)

    def num_children(self, p):
        '''Return the number of children of Position p.'''
        node = self._validate(p)
        count = 0
        if node._left is not None:
            count += 1
        if node._right is not None:
            count += 1
        return count

    def root(self):
        '''Return the root Position of the tree.'''
        return self._make_position(self._root)

    def _add_root(self, e):
        '''Place the element e as the root of an empty tree and return new Position.'''
        if self._root is not None: raise ValueError('Root exists')
        self._size = 1
        self._root = self._Node(e)
        return self._make_position(self._root)

    def _add_left(self, p, e):
        '''Create a new left child for Position p, storing element e. Return the Position of new node. ...'''
        node = self._validate(p)
        if node._left is not None: raise ValueError('Left child exists')
        self._size += 1
        node._left = self._Node(e, node)    # node is its parent
        return self._make_position(node._left)

    def _add_right(self, p, e):
        node = self._validate(p)
        if node._right is not None: raise ValueError('Right child exists.')
        self._size += 1
        node._right = self._Node(e, node)
        return self._make_position(node._right)

    def _replace(self, p, e):
        '''Replace the element at position p with e, and return old element.'''
        node = self._validate(p)
        old = node._element
        node._element = e
        return old

    def _delete(self, p):
        '''Delete the node at Position p, and replace it with its child, if any.
        Return the element that had been stored at Position p. 
        Raise ValueEoor if Position p is valid or p has two children.323'''
        pass
    
    def _attach(self, p, t1, t2):
        pass







        
    
        

        

        


