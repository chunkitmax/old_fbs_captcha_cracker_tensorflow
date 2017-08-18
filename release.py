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
		if np.any(img):
			verify_text = recognizer.recognize(img)
		retObj = { "verify_text": verify_text }
		if headers and headers['Set-Cookie']:
			retObj['cookie'] = headers['Set-Cookie']
		self.wfile.write(bytes(json.dumps(retObj), 'UTF-8'))

def setup():
	global recognizer
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
