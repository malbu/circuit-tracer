# serve_graph.py
import signal, time
from circuit_tracer.frontend.local_server import serve

GRAPH_DIR = "graphs"          # the folder that holds *.json and graph-metadata.json
PORT      = 8046              # pick any free port

server = serve(data_dir=GRAPH_DIR, port=PORT)

print(f"Server running at http://localhost:{PORT}/index.html?slug=smiles-fromprompt")
print("Press Ctrl-C to stop…")

try:                          # keep the process alive until you interrupt it
    signal.pause()
except KeyboardInterrupt:
    pass
finally:
    server.stop()