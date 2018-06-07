import socket

class Connection:

    def __init__(self, sock=None):
        if sock is None:
            self.sock = socket.socket()
        else:
            self.sock = sock
        self._valid = True

    def connect(self, host, port):
        self.sock.connect((host, port))

    def send(self, msg):
        print('sending msg of length', len(msg))
        i = (str(len(msg)).rjust(10)).encode()
        self.sock.send(i)
        totalsent = 0
        while totalsent < len(msg):
            sent = self.sock.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
                self._valid = False
                return
            totalsent = totalsent + sent

    def receive(self):
        msglen = int(self.sock.recv(10))
        print('receiving msg of length', msglen)
        chunks = []
        bytes_recd = 0
        while bytes_recd < msglen:
            chunk = self.sock.recv(msglen - bytes_recd)
            if chunk == '':
                raise RuntimeError("socket connection broken")
                self._valid = False
                return None
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
            print('bytes_recd', bytes_recd)
        print('receive ended')
        return b''.join(chunks)

    def is_valid(self):
        return self._valid
