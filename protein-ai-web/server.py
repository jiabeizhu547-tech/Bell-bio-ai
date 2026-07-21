"""
Protein AI Local Server 鈥?serves landing page + runs ML inference directly.

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
import sys as _sys
import os as _os
try:
    _sys.stderr.flush()
except (OSError, AttributeError):
    _sys.stderr = open(_os.devnull, 'w')
_os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

# Ensure script dir is in sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

PORT = int(os.environ.get("PORT", "8765"))
HOST = os.environ.get("HOST", "127.0.0.1")
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

# Lazy import — models load on first API call
_api = None
def get_api():
    global _api
    if _api is None:
        from api_backend import predict_ss, predict_ec, predict_mutation, predict_ss_batch
        _api = {"predict_ss": predict_ss, "predict_ec": predict_ec, "predict_mutation": predict_mutation, "predict_ss_batch": predict_ss_batch}
    return _api


def _preload_models():
    import threading
    def _load():
        try:
            from api_backend import predict_ss, predict_ec
            predict_ss('MKVLILACLVALALACTVQA')
            predict_ec('MKVLILACLVALALACTVQA')
            print('[server] Models loaded')
        except:
            pass
    t = threading.Thread(target=_load, daemon=True)
    t.start()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=LOCAL_DIR, **kwargs)

    # ---- Routing ----
    def do_GET(self):
        if ".." in self.path:
            self.send_error(403)
            return
        if self.path.startswith("/static/"):
            self._serve_static()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path in ("/api/predict_ss", "/api/predict_ec", "/api/predict_mutation", "/api/predict_ss_batch"):
            self._handle_api(self.path)
        else:
            self.send_error(404, "Not found")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # ---- Handlers ----
    def _serve_static(self):
        path = self.path.lstrip("/")
        filepath = os.path.normpath(os.path.join(LOCAL_DIR, path))
        if not filepath.startswith(os.path.normpath(LOCAL_DIR)):
            self.send_error(403)
            return
        if not os.path.exists(filepath):
            self.send_error(404)
            return
        self.send_response(200)
        ct = "image/png" if path.endswith(".png") else "application/octet-stream"
        self.send_header("Content-Type", ct)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())

    def _handle_api(self, path):
        try:
            content_length = min(int(self.headers.get("Content-Length", "0")), 1048576)  # Max 1MB
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
            elif path == "/api/predict_ss_batch":
                result = api["predict_ss_batch"](data.get("fasta", ""))
            elif path == "/api/predict_mutation":
                result = api["predict_mutation"](data.get("sequence", ""), data.get("mutations", ""))
            else:
                self._json_response(404, {"error": "Unknown endpoint"})
                return
        except Exception as e:
            result = {"error": "Prediction failed"}

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
        if not args:
            return
        msg = str(args[0])
        if "POST /api/" in msg:
            print(f"[api] {msg}")
        elif self.path.startswith("/static/"):
            return
        elif "Exception" in msg or "Traceback" in msg:
            return
        else:
            print(f"[server] {msg}")


if __name__ == "__main__":
    print(f"""
==================================================
  Protein AI - Local Server
  Open:  http://{HOST}:{PORT}
  ML inference runs in-process (no Gradio needed)
  Press Ctrl+C to stop
==================================================
""")
    with http.server.HTTPServer((HOST, PORT), Handler) as httpd:
        _preload_models()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[server] Shutting down...")
            httpd.shutdown()

