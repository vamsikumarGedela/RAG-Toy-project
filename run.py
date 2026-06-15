import threading
import time
import webbrowser
import urllib.request
import uvicorn


def _open_browser():
    # Open browser as soon as uvicorn is up — loading screen handles the rest
    for _ in range(30):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/api", timeout=1)
            webbrowser.open("http://127.0.0.1:8000")
            return
        except Exception:
            time.sleep(1)


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
