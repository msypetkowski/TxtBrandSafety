#!/usr/bin/env python3
import sys
from agents import WorkerAgent

if __name__ == '__main__':
    if len(sys.argv) == 2:
        WorkerAgent(sys.argv[1]).main_loop()
    else:
        print('Server address not given')
