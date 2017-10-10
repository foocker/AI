from LinkedQueue import LinkedQueue

class Tree:
    '''Abstract base class representing a tree structure.'''
    #-----nested Position class-----
    class Position:
        '''An abstraction representing the location of a single element.'''

        def element(self):
            '''Return the element stored at this Positon.'''
            raise NotImplementedError('must be implemented by subclass')
            
        def __eq__(self, other):
            raise NotImplementedError('must be implemented by subclass')
        
        def __ne__(self, other):
            return not(self == other)

    #-----abstract metheods that concrete subclass must support-----
    def root(self):
        '''Return Position representing the tree s root (or None if empty).'''
        raise NotImplementedError('must be implemented by subclass')
    
    def parent(self):
        '''Return Position representing p s parent (or None if p is root).'''
        raise NotImplementedError('must be implemented by subclass')
    
    def num_children(self):
        '''Return the number of children that Position p has.'''
        raise NotImplementedError('must be implemented by subclass')
    
    def children(self, p):
        '''Generate an iteration of Positions representing p s children.'''
        raise NotImplementedError('must be implemented by subclass')
    
    def __len__(self):
        '''Return the total number of elements in the tree.'''
        raise NotImplementedError('must be implemented by subclass')
    
    #-----concrete methods implemented in this class-----
    def is_root(self, p):
        return self.root() == p
    
    def is_leaf(self, p):
        return self.num_children(p) == 0

    def is_empty(self):
        return len(self) == 0
    
    def depth(self, p):
        '''Return the number of levels separating Position p from the root'''
        if self.is_root(p):
            return 0
        else:
            return 1 + self.depth(self.parent(p))

    def _height1(self):
        '''Return the height of the tree.'''
        return max(self.depth(p) for p in self.positions() if self.is_leaf(p))

    def _height2(self, p):
        '''Return the height of the subtree rooted at Position p.'''
        if self.is_leaf(p):
            return 0
        else:
            return 1 + max(self._height2(c) for c in self.children(p))

    def height(self, p=None):
        '''Return the height of the subtree rooted at Position p.
        If p is None, return the height of the entire tree.'''
        if p is None:
            p = self.root()
        return self._height2(p)

    def __iter__(self):
        '''Generate an iteration of the tree's element.'''
        for p in self.positions():    # use same order as positios()
            yield p.element()

    def preorder(self):
        '''Generate a preorder iteration of positions in the tree.'''
        if not self.empty():
            for p in self._subtree_preorder(self.root()):    # start recursion
                yield p
    
    def _subtree_preorder(self, p):
        '''Generate a preorder iteration of positions in subtree rooted at p.'''
        yield p    # visit p before its subtree
        for c in self.children(p):    # for each child c
            for other in self._subtree_preorder(c):    # for preoder of c's subtree
                yield other    # yielding each to our caller
    
    def positions(self):
        '''Generate an iteration of the tree's positions.'''
        return self.preorder()

    '''An implementation of the positions method for the Tree class
that relies on a preorder traversal to generate the results.'''
    
    def postorder(self):
        if not self.is_empty():
            for p in self._subtree_postorder(self.root()):    # start recursion
                yield p
    
    def _subtree_postorder(self, p):
        for c in self.children(p):    # for each child c
            for other in self._subtree_postorder(c):    # do postorder of c's subtree
                yield other    # yielding each to our caller
        yield p    #visit p after its subtrees

    def breadthfirst(self):
        '''Generate a breadtg-first iteration of the positions of tree.'''
        if not self.is_empty():
            fringe = LinkedQueue()    # known positions not yet yielded
            fringe.enqueue(self.root())    #starting with the root
            while not fringe.is_empty():
                p = fringe.dequeue()    # remove from front of queue
                yield p                 # report this position
                for c in self.children(p):
                    fringe.enqueue(c)    # add children to back of queue
    

if __name__ == '__main__':
    print('kiil.')




