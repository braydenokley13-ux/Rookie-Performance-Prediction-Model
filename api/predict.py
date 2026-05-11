"""Vercel serverless function: POST /api/predict"""

import json
from http.server import BaseHTTPRequestHandler

from _prediction import run_prediction


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            raw = self.rfile.read(length) if length > 0 else b'{}'
            stats = json.loads(raw.decode('utf-8') or '{}')
        except json.JSONDecodeError as exc:
            self._send_json(400, {'success': False, 'error': f'Invalid JSON: {exc}'})
            return

        try:
            result = run_prediction(stats)
            self._send_json(200, result)
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {'success': False, 'error': str(exc)})

    def do_GET(self):
        self._send_json(405, {'success': False, 'error': 'Use POST with JSON stats.'})
