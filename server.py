# server.py
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi import Form
import multiprocessing
import threading
import os
import app as ml 
import datetime

API_TOKEN = "uBJjvkPOIFJguPO" #auth key
app = FastAPI(title="KRS Door Automation", version="1.0")

manager = multiprocessing.Manager()
live_process: multiprocessing.Process = None
stop_live_flag = manager.Event()
last_detection = manager.dict()
live_lock = threading.Lock()

def verify_token(x_api_key: str = Header(...)):
    if x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

def live_worker(shared_last_detection, stop_event):

    ml.start_live_check(
        show_window=True,
        shared_last_detection=shared_last_detection,
        stop_flag=stop_event
    )

@app.get("/heartbeat")
def heartbeat(x_api_key: str = Header(...)):
    verify_token(x_api_key)
    return {"status": "live"}

@app.get("/faces")
def list_faces(x_api_key: str = Header(...)):
    verify_token(x_api_key)
    return {"faces": list(ml.encodeDict.keys())}

@app.post("/live/start")
def start_live(x_api_key: str = Header(...)):
    verify_token(x_api_key)
    global live_process, stop_live_flag

    with live_lock:
        if live_process is not None and live_process.is_alive():
            return {"status": "already_running"}

        stop_live_flag.clear()
        live_process = multiprocessing.Process(
            target=live_worker,
            args=(last_detection, stop_live_flag)
        )
        live_process.start()
        return {"status": "started"}

@app.post("/live/stop")
def stop_live(x_api_key: str = Header(...)):
    verify_token(x_api_key)
    global live_process, stop_live_flag

    with live_lock:
        if live_process is None or not live_process.is_alive():
            return {"status": "not_running"}

        stop_live_flag.set()
        live_process.join(timeout=5)
        if live_process.is_alive():
            live_process.terminate()
        live_process = None
        return {"status": "stopped"}

@app.get("/live/status")
def live_status(x_api_key: str = Header(...)):
    verify_token(x_api_key)
    with live_lock:
        detection = dict(last_detection)

    if not detection:
        return {"status": "no data yet"}

    if detection.get("bbox"):
        detection["bbox"] = [int(x) for x in detection["bbox"]]
    if "label" in detection:
        detection["label"] = int(detection["label"])
    return detection

@app.get("/logs")
def get_logs(x_api_key: str = Header(...)):
    verify_token(x_api_key)
    if not hasattr(ml, "LOG_FILE") or not ml.LOG_FILE or not os.path.exists(ml.LOG_FILE):
        return {"log": []}
    with open(ml.LOG_FILE, "r") as f:
        return {"log": f.readlines()}

if __name__ == "__main__":
    print("[INFO] Starting API at http://127.0.0.1:8000")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
