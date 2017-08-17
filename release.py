from http.server import BaseHTTPRequestHandler, HTTPServer
import recognizer
import numpy as np
import requests
import cv2
import json

PORT = 8080
recognizer = recognizer.Recognizer()

class MyHandler(BaseHTTPRequestHandler):
    
    def url_to_image(self, url, headers = None):
        res = requests.get(url, headers = headers)
        image = np.asarray(bytearray(res.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image, res.headers

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        img, headers = self.url_to_image('https://w6.ab.ust.hk/fbs_user/Captcha.jpg')
        verify_text = None
        if img != None:
            verify_text = recognizer.recognize(img)
        self.wfile.write(bytes(json.dumps({ "verify_text": verify_text, "headers": headers }), 'UTF-8'))

def setup():
    httpd = HTTPServer(('localhost', PORT), MyHandler)
    try:
        print('Server starts')
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    del recognizer
    print('Server stops')

if __name__ == "__main__":
    setup()
