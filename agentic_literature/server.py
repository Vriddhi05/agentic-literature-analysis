"""Local HTTP interface; uses only Python's standard library."""
from __future__ import annotations
import json, mimetypes, os, re
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from .pipeline import Source, analyze

MAX_BYTES = 15 * 1024 * 1024
ROOT = Path(__file__).resolve().parent.parent

def parse_uploads(body: bytes, content_type: str) -> list[Source]:
    message = BytesParser(policy=default).parsebytes((f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body))
    sources=[]
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data": continue
        name=part.get_param("name", header="content-disposition")
        filename=part.get_filename()
        if name != "files" or not filename: continue
        data=part.get_payload(decode=True) or b""
        safe=Path(filename.replace("\\", "/")).name[:120]
        suffix=Path(safe).suffix.lower()
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
                import io
                pdf=PdfReader(io.BytesIO(data), strict=False)
                pages=[(p.extract_text() or "") for p in pdf.pages]
            except ImportError as e:
                raise ValueError("PDF support requires pypdf. Run: pip install -r requirements.txt") from e
            except Exception as e:
                raise ValueError(f"Could not read {safe}: {e}") from e
        elif suffix in {".txt", ".md"}:
            pages=[data.decode("utf-8-sig", errors="replace")]
        else: raise ValueError(f"Unsupported file type for {safe}. Use PDF, TXT, or Markdown.")
        sources.append(Source(safe, pages))
    return sources

class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload, content_type="application/json; charset=utf-8"):
        body=payload if isinstance(payload, bytes) else (json.dumps(payload, ensure_ascii=False).encode() if content_type.startswith("application/json") else payload.encode())
        self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.send_header("X-Content-Type-Options", "nosniff"); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        path=urlparse(self.path).path
        if path == "/api/health": return self._send(200, {"status":"ok", "pipeline":"offline-extractive"})
        if path in {"/", "/index.html"}: return self._send(200, (ROOT/"static"/"index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/app.js": return self._send(200, (ROOT/"static"/"app.js").read_bytes(), "text/javascript; charset=utf-8")
        if path == "/styles.css": return self._send(200, (ROOT/"static"/"styles.css").read_bytes(), "text/css; charset=utf-8")
        self._send(404, {"error":"Not found"})
    def do_POST(self):
        if urlparse(self.path).path != "/api/analyze": return self._send(404,{"error":"Not found"})
        try:
            length=int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BYTES: return self._send(413,{"error":"Upload must be between 1 byte and 15 MB."})
            sources=parse_uploads(self.rfile.read(length), self.headers.get("Content-Type", ""))
            result=analyze(sources)
            self._send(200,result)
        except ValueError as e: self._send(400,{"error":str(e)})
        except Exception as e: self.log_error("analysis failed: %s",e); self._send(500,{"error":"Analysis failed. Check the server log."})
    def log_message(self, fmt, *args): print("[%s] %s"%(self.log_date_time_string(), fmt%args))

def main():
    host=os.getenv("HOST", "127.0.0.1"); port=int(os.getenv("PORT", "8000"))
    server=ThreadingHTTPServer((host, port), Handler)
    print(f"Agentic AI Framework for Literature Analysis running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nStopping server...")
    finally: server.server_close()
if __name__ == "__main__": main()
