# mcp_proxy.py
import sys, json, httpx

REMOTE = "http://ip_address:8000/mcp/eval_service"

def main():
    request = {"parameters": {"output_dir": "/path/to/your/output/dir"}}
    # req = json.load(sys.stdin)
    resp = httpx.post(REMOTE, json=request).json()
    json.dump(resp, sys.stdout)

if __name__ == "__main__":
    main()
