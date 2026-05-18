import subprocess
import time

process = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(5)

!curl -s http://localhost:11434

import requests
import json

def pull_model(model_name):
    print(f"Pulling {model_name}...")
    response = requests.post(
        "http://localhost:11434/api/pull",
        json={"model": model_name, "stream": True},
        stream=True,
        timeout=1800,
    )
    last_status = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            status = data.get("status", "")
            if status != last_status and "pulling" not in status.lower():
                print(f"  {status}")
                last_status = status
    print(f"  ✓ {model_name} done.\n")


pull_model("qwen3.5:9b")
