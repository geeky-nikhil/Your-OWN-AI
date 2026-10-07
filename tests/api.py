import json, os, socket, subprocess, tempfile, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def free_port():
    with socket.socket() as s: s.bind(('127.0.0.1',0)); return s.getsockname()[1]
class Ollama(BaseHTTPRequestHandler):
    prompts=[]; generates=0
    def log_message(self,*args): pass
    def do_GET(self): self.reply({'models':[]})
    def reply(self,data):
        self.send_response(200); self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(data).encode())
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path=='/api/embeddings' and 'fail' in body['prompt']:
            self.send_response(503);self.end_headers();return
        if self.path=='/api/embeddings': self.reply({'embedding':[0.,1.] if 'unrelated' in body['prompt'] else [1.,0.]})
        else:
            if 'generation-error' in body['prompt']:
                self.send_response(503);self.end_headers();return
            Ollama.generates+=1;Ollama.prompts.append(body['prompt']);self.reply({'response':'The answer is 42 [1].'})
server=ThreadingHTTPServer(('127.0.0.1',0),Ollama);threading.Thread(target=server.serve_forever,daemon=True).start()
port=free_port();base=f'http://127.0.0.1:{port}'
def request(path,body=None,method=None):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(base+path,data,{'Content-Type':'application/json'},method=method)
    try:
        with urllib.request.urlopen(req,timeout=10) as r:return r.status,json.loads(r.read())
    except urllib.error.HTTPError as e:return e.code,json.loads(e.read())
def start(folder):
    env=dict(os.environ,PORT=str(port),DATA_DIR=folder,OLLAMA_HOST='127.0.0.1',OLLAMA_PORT=str(server.server_port))
    p=subprocess.Popen(['./build/app'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    for _ in range(100):
        if p.poll() is not None:raise RuntimeError(p.stderr.read().decode())
        try: request('/status');return p
        except (OSError,urllib.error.URLError):time.sleep(.05)
    p.terminate();raise RuntimeError('startup timeout')
def stop(p):p.terminate();p.wait(timeout=10)
with tempfile.TemporaryDirectory() as folder:
    p=start(folder)
    try:
        v=','.join(['1']*16)
        assert request('/search?v='+v+'&k=0')[0]==400
        assert request('/search?v='+v+'&metric=invalid')[0]==400
        assert request('/search?v='+v+'&efSearch=0')[0]==400
        for metric in ['cosine','euclidean','manhattan']:
            assert request('/search?v='+v+'&metric='+metric)[0]==200
        assert request('/doc/insert',{'title':'x','text':'words','chunkWords':3,'overlapWords':3})[0]==400
        status,d=request('/doc/insert',{'title':'answer','text':'The answer is 42.','chunkWords':3,'overlapWords':1});assert status==200 and d['chunks']==2
        status,d=request('/doc/ask',{'question':'what is the answer?','k':3});assert status==200 and '[1]' in d['answer'] and len(d['contexts'])==2
        assert 'Do not use outside knowledge' in Ollama.prompts[-1]
        assert request('/doc/ask',{'question':'generation-error'})[0]==503
        assert request('/doc/insert',{'title':'failed batch','text':'first good then fail','chunkWords':2,'overlapWords':0})[0]==503
        assert request('/status')[1]['docCount']==2
        before=Ollama.generates
        status,d=request('/doc/ask',{'question':'unrelated','maxDistance':.1});assert status==200 and d['abstained'] and d['contexts']==[]
        assert Ollama.generates==before
        assert request('/doc/search',{'question':'x','maxDistance':-1})[0]==400
        status,item=request('/insert',{'metadata':'persistent','category':'test','embedding':[1.]*16});assert status==200
        assert request('/status')[1]['demoCount']==21
    finally:stop(p)
    p=start(folder)
    try:
        assert request('/status')[1]['demoCount']==21
        assert len(request('/doc/list')[1])==2
        assert request('/delete/'+str(item['id']),method='DELETE')[1]['ok']
    finally:stop(p)
server.shutdown();print('API, grounded prompt, no-context abstention, and restart checks passed (mock Ollama)')
