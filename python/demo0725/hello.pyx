cdef extern from "ha.h":
    void hello()

def say_hello():
    hello()