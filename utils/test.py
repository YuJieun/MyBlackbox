from socket import *
from sys import exit
from data_file import SocketInfo

class SocketInfo(SocketInfo):
    HOST='127.0.0.1'

csock=socket(AF_INET,SOCK_STREAM)
csock.connect(SocketInfo.ADDR)

while True:
    try:
        commend = raw_input(">>")
        csock.send(commend)
        print('wait')
        commend = csock.recv(SocketInfo.BUFSIZE)
        print(commend)

    except Exception as e:
        print('fail')
        csock.close()
        exit()