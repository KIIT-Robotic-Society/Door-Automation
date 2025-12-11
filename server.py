# fastAPI server for face recognition system with REST API endpoints
import uvicorn
from fastapi import FastAPI, Header, HTTPException
import multiprocessing
import threading
import os
import time
import app as ml  # import the main ML face recognition module

# API authentication token
API_TOKEN = "uBJjvkPOIFJguPO"  # Auth key for securing endpoints

# initialize FastAPI application
app = FastAPI(title="KRS Door Automation", version="1.0")

# multiprocessing manager for sharing data between processes
manager = multiprocessing.Manager()
live_process: multiprocessing.Process = None  # process running live camera feed
stop_live_flag = manager.Event()  # event to signal process termination
last_detection = manager.dict()  # shared dictionary for detection results
live_lock = threading.Lock()  # lock to prevent race conditions when starting/stopping

def verify_token(x_api_key: str = Header(...)):
    """Verify API token from request header"""
    if x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

def live_worker(shared_last_detection, stop_event):
    """Worker function to run face recognition in separate process"""
    import sys
    import traceback
    
    print("[DEBUG] Live worker process started", flush=True)
    
    # Signal initialization start immediately
    shared_last_detection["status"] = "initializing"
    print("[DEBUG] Status set to 'initializing'", flush=True)
    
    try:
        print("[DEBUG] Calling ml.start_live_check()...", flush=True)
        # Run the live check - this will update status to "ready" then "running"
        ml.start_live_check(
            show_window=True,
            shared_last_detection=shared_last_detection,
            stop_flag=stop_event
        )
        print("[DEBUG] ml.start_live_check() completed", flush=True)
    except Exception as e:
        print(f"[ERROR] Live worker crashed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        shared_last_detection["status"] = "error"
        shared_last_detection["error"] = str(e)

@app.get("/heartbeat")
def heartbeat(x_api_key: str = Header(...)):
    """Health check endpoint to verify API is running"""
    verify_token(x_api_key)
    return {"status": "live"}

@app.get("/faces")
def list_faces(x_api_key: str = Header(...)):
    """Get list of all registered face names"""
    verify_token(x_api_key)
    return {"faces": list(ml.encodeDict.keys())}

@app.post("/live/start")
def start_live(x_api_key: str = Header(...)):
    """Start live camera feed for face recognition"""
    verify_token(x_api_key)
    global live_process, stop_live_flag
    
    print("[DEBUG] /live/start endpoint called", flush=True)
    
    with live_lock:
        # check if process is already running
        if live_process is not None and live_process.is_alive():
            print("[DEBUG] Process already running", flush=True)
            return {"status": "already_running"}
        
        # Reset shared state before starting
        print("[DEBUG] Clearing last_detection dict", flush=True)
        last_detection.clear()
        last_detection["status"] = "starting"
        
        # start new recognition process
        stop_live_flag.clear()
        print("[DEBUG] Creating multiprocessing.Process...", flush=True)
        live_process = multiprocessing.Process(
            target=live_worker,
            args=(last_detection, stop_live_flag)
        )
        print("[DEBUG] Starting process...", flush=True)
        live_process.start()
        print(f"[DEBUG] Process started with PID: {live_process.pid}", flush=True)
        
        # Give process a moment to set status
        time.sleep(0.2)
        
        current_status = last_detection.get("status", "unknown")
        print(f"[DEBUG] After 0.2s, status is: {current_status}", flush=True)
        
        return {"status": "started", "pid": live_process.pid, "initial_status": current_status}

@app.post("/live/stop")
def stop_live(x_api_key: str = Header(...)):
    """Stop the live camera feed gracefully"""
    verify_token(x_api_key)
    global live_process, stop_live_flag
    
    with live_lock:
        # check if process exists
        if live_process is None or not live_process.is_alive():
            last_detection.clear()
            return {"status": "not_running"}
        
        # signal process to stop and wait for graceful shutdown
        stop_live_flag.set()
        live_process.join(timeout=5)
        
        # force terminate if still running after timeout
        if live_process.is_alive():
            print("[WARN] Force terminating live process")
            live_process.terminate()
            live_process.join(timeout=2)
        
        live_process = None
        last_detection.clear()
        return {"status": "stopped"}

@app.get("/live/status")
def live_status(x_api_key: str = Header(...)):
    """Get current status of live recognition"""
    verify_token(x_api_key)
    
    with live_lock:
        # Return current state of shared dictionary
        detection = dict(last_detection)
        
        # If no data yet, return appropriate status
        if not detection:
            return {"status": "idle"}
        
        return detection

@app.get("/logs")
def get_logs(x_api_key: str = Header(...)):
    """Retrieve recognition event logs"""
    verify_token(x_api_key)
    
    # check if log file exists
    if not hasattr(ml, "LOG_FILE") or not ml.LOG_FILE or not os.path.exists(ml.LOG_FILE):
        return {"log": []}
    
    # read and return all log entries
    try:
        with open(ml.LOG_FILE, "r") as f:
            return {"log": f.readlines()}
    except Exception as e:
        print(f"[ERROR] Failed to read logs: {e}")
        return {"log": [], "error": str(e)}

@app.get("/debug")
def debug_info(x_api_key: str = Header(...)):
    """Debug endpoint to check system state"""
    verify_token(x_api_key)
    global live_process
    
    return {
        "process_alive": live_process.is_alive() if live_process else False,
        "process_pid": live_process.pid if live_process else None,
        "last_detection": dict(last_detection),
        "encodings_count": len(ml.encodeDict),
        "encoding_names": list(ml.encodeDict.keys()),
        "stop_flag_set": stop_live_flag.is_set(),
        "models_initialized": hasattr(ml, 'anti_spoof_model') and ml.anti_spoof_model is not None
    }

if __name__ == "__main__":
    print("[INFO] Starting API at http://127.0.0.1:8000")
    print("[INFO] Encodings loaded:", len(ml.encodeDict))
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)