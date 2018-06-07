import sys

class WorkerAgent:

    def __init__ (self):
        pass

    def join_to_front_agent(self, addr):
        pass

if __name__ == '__main__':
    if len(sys.argv) == 2:
        WorkerAgent().join_to_front_agent(sys.argv[1])
    else:
        print('Server address not given')
