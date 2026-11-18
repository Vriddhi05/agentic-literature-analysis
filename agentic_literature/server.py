"""Local HTTP interface backed by Python's standard library."""
from __future__ import annotations
import json, os, tempfile, urllib.parse, urllib.request, uuid
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .pipeline import MODEL_CACHE, MODEL_ID, EncoderLoadError, Source, analyze, model_is_loaded

MAX_BYTES=15*1024*1024
ROOT=Path(__file__).resolve().parent.parent
RUNS_DIR=Path(os.getenv("RUNS_DIR",str(ROOT/"data"/"runs")))
FIXTURES_FILE=ROOT/"fixtures"/"papers.json"

def load_fixtures()->list[Source]:
    data=json.loads(FIXTURES_FILE.read_text(encoding="utf-8"));sources=[]
    for paper in data["papers"]:
        metadata={k:v for k,v in paper.items() if k!="text"}
        sources.append(Source(paper["source_id"],[paper["text"]],metadata))
    return sources

def parse_uploads(body:bytes,content_type:str)->list[Source]:
    message=BytesParser(policy=default).parsebytes((f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()+body));sources=[]
    for part in message.iter_parts():
        if part.get_content_disposition()!="form-data" or part.get_param("name",header="content-disposition")!="files":continue
        filename=part.get_filename()
        if not filename:continue
        data=part.get_payload(decode=True) or b"";safe=Path(filename.replace("\\","/")).name[:120];suffix=Path(safe).suffix.lower()
        if suffix==".pdf":
            try:
                from pypdf import PdfReader
                import io
                pdf=PdfReader(io.BytesIO(data),strict=False);pages=[p.extract_text() or "" for p in pdf.pages]
            except ImportError as e:raise ValueError("PDF support requires pypdf. Run: pip install -r requirements.txt") from e
            except Exception as e:raise ValueError(f"Could not read {safe}: {e}") from e
        elif suffix in {".txt",".md"}:pages=[data.decode("utf-8-sig",errors="replace")]
        else:raise ValueError(f"Unsupported file type for {safe}. Use PDF, TXT, or Markdown.")
        sources.append(Source(safe,pages))
    return sources

def save_run(result:dict)->str:
    RUNS_DIR.mkdir(parents=True,exist_ok=True);run_id=uuid.uuid4().hex
    result["run_id"]=run_id;result["created_at"]=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    target=RUNS_DIR/(run_id+".json")
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=RUNS_DIR,delete=False) as f:json.dump(result,f,ensure_ascii=False);temp=Path(f.name)
    temp.replace(target);return run_id

def load_run(run_id:str)->dict:
    if not run_id.isalnum() or len(run_id)!=32:raise FileNotFoundError
    path=RUNS_DIR/(run_id+".json")
    if not path.is_file():raise FileNotFoundError
    return json.loads(path.read_text(encoding="utf-8"))

def run_summary(run:dict)->dict:
    return {"run_id":run["run_id"],"created_at":run["created_at"],"sources":[p["source"] for p in run.get("papers",[])],"status":run.get("quality_review",{}).get("citation_check"),"model_id":run.get("encoder",{}).get("model_id"),"dimensions":run.get("encoder",{}).get("dimensions")}

def format_ref(ref:dict)->str:
    location=ref.get("location") or (f"p. {ref['page']}" if ref.get("page") else "source text")
    url=ref.get("url")
    origin=f"[{ref.get('source','source')}, {location}]({url})" if url else f"{ref.get('source','source')}, {location}"
    return f"{origin}: {ref.get('quote','')}"

def markdown_report(run:dict)->str:
    encoder=run.get("encoder",{});cluster=run.get("clusters",{});quality=run.get("quality_review",{})
    out=["# Literature Analysis Report","",f"Run: {run.get('run_id','')}",f"Created: {run.get('created_at','')}",f"Selected encoder: {encoder.get('provider','unknown')} / {encoder.get('model_id','unknown')} ({encoder.get('dimensions','?')} dimensions)","", "## Overview",run["synthesis"]["overview"],"","## Clustering",json.dumps(cluster.get("settings",{}),ensure_ascii=False),f"Status: {cluster.get('status','unknown')}","","## Themes"]
    out += [f"- **{t['theme']}** ({t['mentions']} mentions; sources: {', '.join(t.get('supporting_source_ids',[])) or 'none'})" for t in run["synthesis"]["themes"]]
    for paper in run["papers"]:
        prov=paper.get("provenance",{});out += ["",f"## {paper['title']}",f"Source: {paper['source']} ({paper.get('text_available','')}; full text included: {paper.get('full_text_included')})"]
        if prov.get("source_url"):out.append(f"Provenance: {prov.get('authors','')}, {prov.get('year','')}, [{prov.get('source_url')}]({prov.get('source_url')})")
        out += ["",paper["summary"],"","Summary references:"]+[f"- {format_ref(x)}" for x in paper["summary_references"]]
    out += ["","## Candidate research gaps"]
    for gap in run["synthesis"]["gaps"]:out += [f"- **{gap['question']}** _Support: {gap['claim_support_status']}_",f"  Basis: {gap['basis']}"]+[f"  - {format_ref(x)}" for x in gap["evidence"]]
    out += ["","## Citation and support audit",f"Reference validity: {quality.get('reference_validity',{}).get('status','unknown')} ({quality.get('reference_validity',{}).get('checked_count',0)} references checked)","Claim support is reported separately from whether a citation resolves to source text.","","## LLM synthesis",f"Status: {run.get('llm_synthesis',{}).get('status','unavailable')}"]
    llm=run.get("llm_synthesis",{});out.append(llm.get("text") or llm.get("reason", "Not run."))
    out += ["","## Limitations"]+[f"- {x}" for x in quality.get("limitations",[])]
    return "\n".join(out)+"\n"

def scholarly_search(query:str)->dict:
    q=" ".join(query.split())[:240]
    if not q:raise ValueError("Provide a search query.")
    url="https://api.crossref.org/works?"+urllib.parse.urlencode({"query.bibliographic":q,"rows":8,"select":"DOI,title,author,published,container-title,URL,type"})
    req=urllib.request.Request(url,headers={"User-Agent":"AgenticLiteratureAnalysis/1.1 (local demo)"})
    with urllib.request.urlopen(req,timeout=12) as response:data=json.loads(response.read())
    items=[]
    for item in data.get("message",{}).get("items",[]):
        date=item.get("published",{}).get("date-parts",[[None]])[0]
        items.append({"title":(item.get("title") or ["Untitled"])[0],"authors":[" ".join([a.get("given",""),a.get("family","")]).strip() for a in item.get("author",[])[:4]],"year":date[0] if date and date[0] else None,"venue":(item.get("container-title") or [None])[0],"doi":item.get("DOI"),"url":item.get("URL"),"type":item.get("type")})
    return {"provider":"Crossref","query":q,"count":len(items),"results":items}

class Handler(BaseHTTPRequestHandler):
    def _send(self,status,payload,content_type="application/json; charset=utf-8",download=None):
        body=payload if isinstance(payload,bytes) else (json.dumps(payload,ensure_ascii=False).encode() if content_type.startswith("application/json") else payload.encode())
        self.send_response(status);self.send_header("Content-Type",content_type);self.send_header("Content-Length",str(len(body)));self.send_header("X-Content-Type-Options","nosniff")
        if download:self.send_header("Content-Disposition",f'attachment; filename="{download}"')
        self.end_headers();self.wfile.write(body)
    def _analyze(self,sources,mode):
        try:
            result=analyze(sources,encoder_mode=mode);save_run(result);return self._send(200,result)
        except EncoderLoadError as e:
            return self._send(503,{"error":"Semantic encoder unavailable.","encoder_error":e.message,"selected_model":MODEL_ID,"fallback_available":True,"fallback_label":"Explicit local lexical fallback","instruction":"Choose the labeled fallback to continue without semantic embeddings."})
        except ValueError as e:return self._send(400,{"error":str(e)})
        except Exception as e:self.log_error("analysis failed: %s",e);return self._send(500,{"error":"Analysis failed. Check the server log."})
    def do_GET(self):
        parsed=urllib.parse.urlparse(self.path);path=parsed.path
        if path=="/api/health":return self._send(200,{"status":"ok","selected_encoder":MODEL_ID,"encoder_loaded":model_is_loaded(),"cache_dir":str(MODEL_CACHE),"fallback_available":True,"llm_status":"configured_unvalidated" if os.getenv("OPENAI_API_KEY") else "unavailable_no_api_key"})
        if path=="/api/fixtures":
            data=json.loads(FIXTURES_FILE.read_text(encoding="utf-8"));return self._send(200,{"collection_title":data["collection_title"],"count":len(data["papers"]),"papers":[{k:p[k] for k in ("source_id","title","authors","year","subtopic","text_available","source_url")} for p in data["papers"]]})
        if path=="/api/runs":
            RUNS_DIR.mkdir(parents=True,exist_ok=True);runs=[]
            for f in sorted(RUNS_DIR.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True):
                try:runs.append(run_summary(json.loads(f.read_text(encoding="utf-8"))))
                except (ValueError,KeyError):continue
            return self._send(200,{"runs":runs})
        if path.startswith("/api/runs/"):
            rest=path[len("/api/runs/"):]
            try:
                if rest.endswith("/report.md"):
                    run=load_run(rest[:-len("/report.md")]);return self._send(200,markdown_report(run),"text/markdown; charset=utf-8",f"literature-analysis-{run['run_id'][:8]}.md")
                if rest.endswith("/report.json"):
                    run=load_run(rest[:-len("/report.json")]);return self._send(200,run,"application/json; charset=utf-8",f"literature-analysis-{run['run_id'][:8]}.json")
                return self._send(200,load_run(rest))
            except FileNotFoundError:return self._send(404,{"error":"Saved run not found"})
        if path=="/api/search":
            try:return self._send(200,scholarly_search(urllib.parse.parse_qs(parsed.query).get("q",[""])[0]))
            except ValueError as e:return self._send(400,{"error":str(e)})
            except Exception as e:return self._send(502,{"error":"Crossref retrieval failed: "+str(e)[:200]})
        if path in {"/","/index.html"}:return self._send(200,(ROOT/"static"/"index.html").read_bytes(),"text/html; charset=utf-8")
        if path=="/app.js":return self._send(200,(ROOT/"static"/"app.js").read_bytes(),"text/javascript; charset=utf-8")
        if path=="/styles.css":return self._send(200,(ROOT/"static"/"styles.css").read_bytes(),"text/css; charset=utf-8")
        return self._send(404,{"error":"Not found"})
    def do_POST(self):
        parsed=urllib.parse.urlparse(self.path);mode=urllib.parse.parse_qs(parsed.query).get("encoder",["semantic"])[0]
        try:length=int(self.headers.get("Content-Length","0"))
        except ValueError:return self._send(400,{"error":"Invalid Content-Length"})
        if length>MAX_BYTES:return self._send(413,{"error":"Request exceeds 15 MB."})
        body=self.rfile.read(length) if length else b""
        if parsed.path=="/api/analyze-fixtures":
            try:
                if body:mode=json.loads(body).get("encoder",mode)
                return self._analyze(load_fixtures(),mode)
            except (ValueError,KeyError,json.JSONDecodeError) as e:return self._send(400,{"error":str(e)})
        if parsed.path=="/api/analyze":
            try:return self._analyze(parse_uploads(body,self.headers.get("Content-Type","")),mode)
            except ValueError as e:return self._send(400,{"error":str(e)})
        return self._send(404,{"error":"Not found"})
    def log_message(self,fmt,*args):print("[%s] %s"%(self.log_date_time_string(),fmt%args))

def main():
    host=os.getenv("HOST","127.0.0.1");port=int(os.getenv("PORT","8000"));server=ThreadingHTTPServer((host,port),Handler)
    print(f"Agentic AI Framework for Literature Analysis running at http://{host}:{port}")
    try:server.serve_forever()
    except KeyboardInterrupt:print("\nStopping server...")
    finally:server.server_close()
if __name__=="__main__":main()
