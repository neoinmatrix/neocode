# coding=utf-8
import socket

obj = socket.socket()

obj.connect(("127.0.0.1",8080))

ret_bytes = obj.recv(1024)
ret_str = str(ret_bytes)
print(ret_str)

while True:
    inp = raw_input("你好请问您有什么问题")
    if inp == "q":
        obj.sendall(bytes(inp))
        break
    else:
        obj.sendall(bytes(inp))
        ret_bytes = obj.recv(1024)
        ret_str = str(ret_bytes)
        print(ret_str)
# raw_input_A = raw_input("raw_input: ")
# print raw_input_A
# input_A = input("Input: ")
# print input_A
