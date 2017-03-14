# coding=utf-8
import os    
import urllib   
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler  
class TestHTTPHandler(BaseHTTPRequestHandler):  
    def do_GET(self):  
        templateStr = '''''   
<html>   
<head>   
<title>QR Link Generator</title>   
</head>   
<body>   
hello world
</body>   
</html> '''  
        self.protocal_version = 'HTTP/1.1' 
        self.send_response(200) 
        self.send_header("Welcome", "Contect")
        self.end_headers()  
        self.wfile.write(templateStr)  
      

def start_server(port):  
    http_server = HTTPServer(('', int(port)), TestHTTPHandler)  
    http_server.serve_forever()

# os.chdir('static')  
start_server(8000) 