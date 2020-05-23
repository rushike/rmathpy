from threading import Thread

class NumberThread(Thread):
    __WORKING__ = 0
    __IN_ACTIVE__ = -1
    __FINAL__ = 1

    """[summary]
    
    Arguments:
        Thread {[type]} -- [description]
    """

    def __init__(self, id, state = -1, handler = False, group = None, target = None, name = None, args = (), kwargs = {}):
        """Constructor
        
        Arguments:
            group {[type]} -- [description]
            target {[type]} -- [description]
            name {[type]} -- [description]
        
        Keyword Arguments:
            state {int} -- state of number thread (default: {-1})
            args {tuple} -- [description] (default: {()})
            kwargs {dict} -- [description] (default: {{}})
        """

        super().__init__(group, target, name, args, kwargs)
        self.id = id
        self.state = state
        self._handler = handler

    
    def run(self):
        # print("Run executing .................................................................................")
        # print(self.__str__())
        try :
            if self._target and  not self._handler :
                # print("No Handler")
                self._target(*self._args, **self._kwargs)
            elif self._target and self._handler :
                # print("Handler")
                thread = (self,)
                self._args = thread.__add__(self._args)
                self._target(*self._args, **self._kwargs)
        finally :
            del self._target, self._args, self._kwargs


    def state_in(self, state = __IN_ACTIVE__):
        """Set the state of NumberThread
        
        Arguments:
            state {int} -- state to be set
        """
        self.state = state

    
    @staticmethod
    def count_alive_worker(thread_arr : list) :
        count = 0
        for t in thread_arr :
            if not t :
                continue
            if t.is_alive() :
                count += 1
        return count

    @staticmethod
    def two_dead_worker(thread_arr : list = []) :
        """[summary]
        
        Keyword Arguments:
            thread_arr {list} -- Thread array, list (default: {[]})
        
        Returns:
            list -- indexes of dead threads
        """
        ind = {}
        i , j = 0, 0
        for t in thread_arr :
            if j == 2 :
                return ind
            if not t :
                continue
            if not t.state == NumberThread.__IN_ACTIVE__ : 
                ind[j] = i
                j += 1
            i += 1
        return ind

    @staticmethod
    def workers_in_state(thread_arr : list, state : int = 1):
        """Finds threads from thread list in given state
        
        Arguments:
            thread_arr {list of NumberThread} -- Thread Array, List
            state_list {list} -- State Array, List
        
        Keyword Arguments:
            state {int} -- Thread working state (default: {1})
        
        Returns:
            list -- [description]
        """
        ind = {}
        i  = 0
        for t in thread_arr :
            if not t :
                continue
            if t.state == state : 
                ind[i] = t.id
            i += 1
        return ind
    
    @staticmethod
    def count_workers_in_state(thread_arr : list, state : int = 1):
        """Finds threads from thread list in given state
        
        Arguments:
            thread_arr {list of NumberThread} -- Thread Array, List
            state_list {list} -- State Array, List
        
        Keyword Arguments:
            state {int} -- Thread working state (default: {1})
        
        Returns:
            int -- count of workers in state
        """
        count = 0
        for t in thread_arr :
            if not t :
                continue
            if t.state == state : 
                count += 1
        return count

    @staticmethod
    def __state_str__(thread_arr : list = []) :
        res = "["
        for t in thread_arr :
            if not t :
                continue
            res += "(" + str(t.id) + ". state : " + str(t.state) + ") "
        return res + "]"
        
    def __str__(self):
        """Format : NumberThread object string definition
        """
        try : 
            result =  "id : " + str(self.id) + ", state : " + str(self.state) + ", handler : " + str(self._handler) 
            result += ", target :" + str(self._target) +"(" + str(self._args)[:9] + ",,," + str(self._kwargs)[:7]
        finally :    
            return result + "\n" 

    def __repr__(self):
        return self.__str__()

    

