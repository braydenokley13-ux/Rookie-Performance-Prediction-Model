"""Vercel serverless function: GET /api/health"""

import json
from http.server import BaseHTTPRequestHandler

from _prediction import get_model


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            model = get_model()
            payload = {
                'status': 'ok',
                'model_loaded': True,
                'fallback_model': bool(model.get('fallback', False)),
                'regression_model': model.get('best_regression_name', 'Unknown'),
                'classification_model': model.get('best_classification_name', 'Unknown'),
            }
            status = 200
        except Exception as exc:  # noqa: BLE001
            payload = {'status': 'error', 'model_loaded': False, 'error': str(exc)}
            status = 500

        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
