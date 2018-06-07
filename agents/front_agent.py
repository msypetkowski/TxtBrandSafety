import random
from threading import Thread, Lock
import socket

from agents.connection import Connection


class WorkerConnection():
    def __init__(self, socket, addr):
        Thread.__init__(self)
        self.addr = addr
        self._conn = Connection(sock=socket)

    def alive(self):
        return self._conn.is_valid()

    def query(self, query_body):
        query_body = query_body.encode()
        print('new query for worker:', self.addr)
        if not self._conn.is_valid():
            print('broken connection')
            return None
        self._conn.send(query_body)
        if not self._conn.is_valid():
            print('broken connection')
            return None
        ret = self._conn.receive()
        if not self._conn.is_valid():
            print('broken connection')
            return None
        print('received', ret)
        return ret


class FrontAgent(Thread):

    def __init__(self):
        Thread.__init__(self)
        self._results = {}
        self._workers = []
        self._stopped = False
        pass

    def run(self):
        sock = socket.socket()
        sock.bind(('', 4321))
        sock.listen()
        self._sock = sock
        while True:
            if self._stopped:
                break
            s, addr = sock.accept()
            print('new worker:', addr)
            self._workers.append(WorkerConnection(s, addr))

    def query(self, query_body):
        while True:
            if self._workers:
                worker = random.choice(self._workers)
                if worker.alive():
                    ret = worker.query(query_body)
                else:
                    ret = None
                    print('removing worker:', worker.addr)
                    self._workers.remove(worker)
                if ret is not None:
                    return ret
            else:
                return None

    def stop(self):
        self._stopped = True
        self._sock.close()
