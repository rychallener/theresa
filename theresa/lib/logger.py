class Logger:
    '''
    Simple logging class to reduce screen clutter when
    multiprocessing under MPI.
    '''
    def __init__(self, rank):
        self.rank = rank
    
    def __call__(self, msg):
        if self.rank == 0:
            print(msg)
        
