"""
Protein AI Local Server — serves landing page + runs ML inference directly.

Usage:
    python server.py
    > Landing page: http://localhost:8765
    > API calls handled in-process (no Gradio needed)
"""
import http.server
import json
import os
import sys
import traceback

PORT = 8765
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

# Lazy import — models load on first API call
_api = None

def get_api():
    global _api
    if _api is None:
        from api_backend import predict_ss, predict_ec, predict_mutation
        _api = {"predict_ss": predict_ss, "predict_ec": predict_ec, "predict_mutation": predict_mutation}
    return _api


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=LOCAL_DIR, **kwargs)

    # ---- Routing ----
    def do_GET(self):
        if self.path.startswith("/static/"):
            self._serve_static()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path in ("/api/predict_ss", "/api/predict_ec", "/api/predict_mutation"):
            self._handle_api(self.path)
        else:
            self.send_error(404, "Not found")

    def do_OPTIONS(self):
        self._cors_headers()
        self.send_response(204)
        self.end_headers()

    # ---- Handlers ----
    def _serve_static(self):
        path = self.path.lstrip("/")
        filepath = os.path.join(LOCAL_DIR, path)
        if not os.path.exists(filepath):
            self.send_error(404)
            return
        self.send_response(200)
        ct = "image/png" if path.endswith(".png") else "application/octet-stream"
        self.send_header("Content-Type", ct)
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())

    def _handle_api(self, path):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length else b"{}"
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._json_response(400, {"error": "Invalid JSON"})
            return

        api = get_api()

        try:
            if path == "/api/predict_ss":
                result = api["predict_ss"](data.get("sequence", ""))
            elif path == "/api/predict_ec":
                result = api["predict_ec"](data.get("sequence", ""))
            elif path == "/api/predict_mutation":
                result = api["predict_mutation"](data.get("sequence", ""), data.get("mutations", ""))
            else:
                self._json_response(404, {"error": "Unknown endpoint"})
                return
        except Exception as e:
            result = {"error": f"Server error: {e}\n{traceback.format_exc()}"}

        self._json_response(200, result)

    def _json_response(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt, *args):
        msg = args[0] if args else ""
        if "POST /api/" in msg:
            print(f"[api] {msg}")
        elif self.path.startswith("/static/"):
            return  # suppress static file noise
        else:
            print(f"[server] {msg}")


if __name__ == "__main__":
    print(f"""
==================================================
  Protein AI - Local Server
  Open:  http://localhost:{PORT}
  ML inference runs in-process (no Gradio needed)
  Press Ctrl+C to stop
==================================================
""")
    with http.server.HTTPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[server] Shutting down...")
            httpd.shutdown()
