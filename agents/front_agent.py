import random
from threading import Thread, Lock
import socket

from agents.connection import Connection


class WorkerConnection:
    def __init__(self, sock, addr):
        super().__init__()
        self.addr = addr
        self._conn = Connection(sock=sock)
        self._lock = Lock()

    def alive(self):
        return self._conn.is_valid()

    def query(self, query_body):
        self._lock.acquire()
        try:
            query_body = query_body
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
        finally:
            self._lock.release()


class FrontAgent(Thread):

    def __init__(self):
        Thread.__init__(self)
        self._results = {}
        self._workers = []
        self._stopped = False
        self._sock = None
        pass

    def run(self):
        sock = socket.socket()
        sock.bind(('', 4321))
        sock.listen()
        self._sock = sock
        while True:
            if self._stopped:
                break
            sock.settimeout(0.5)
            try:
                s, addr = sock.accept()
            except socket.timeout:
                if self._stopped:
                    break
                continue
            print('new worker:', addr)
            self._workers.append(WorkerConnection(s, addr))
        self._sock.close()

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
