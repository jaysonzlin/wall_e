# Wall-E Robot Arm

This project provides a simple FastAPI backend, structured as a Claude MCP server, to trigger a `lerobot` evaluation script via an API call.

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository_url>
    cd wall_e
    ```

2.  **Create a Python virtual environment:**
    This isolates the project's dependencies.
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    -   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Ensure Conda is installed and the `lerobot` environment exists.** The backend relies on `conda run -n lerobot ...` to execute the evaluation script.

## Running the Server

1.  **Activate the virtual environment (if not already active):**
    ```bash
    source venv/bin/activate # Or .\venv\Scripts\activate on Windows
    ```

2.  **Start the FastAPI server:**
    ```bash
    python main.py 
    ```
    The server will start and listen on `http://0.0.0.0:8000` by default, making it accessible on your network.

## Using the MCP Server

This server follows the Claude MCP (Multi-Component Protocol) specification.

### Health Check

You can check if the server is running by accessing the `/up` endpoint:

```bash
curl http://<server_ip>:8000/up 
```
Replace `<server_ip>` with the IP address of the machine running the server (use `127.0.0.1` if running locally).

### Invoking the Tool

Send a POST request to the `/invoke` endpoint with a JSON body specifying the `tool_name` as `run_lerobot_eval` and the `inputs` dictionary containing the `output_dir`.

**Example using `curl`:**

```bash
curl -X POST http://<server_ip>:8000/invoke \
-H "Content-Type: application/json" \
-d '{
    "tool_name": "run_lerobot_eval",
    "inputs": {
        "output_dir": "/path/to/your/lerobot/output/directory"
    }
}'
```

Replace `<server_ip>` with the server's IP address and `/path/to/your/lerobot/output/directory` with the actual path.

The server will execute the `conda run -n lerobot python ~/lerobot/lerobot/scripts/eval.py --policy.path=<output_dir>/checkpoints/last/pretrained_model` command.

**Response Format:**

The response will be a JSON object with an `outputs` key containing the execution results:

*   **Success:**
    ```json
    {
        "outputs": {
            "message": "Evaluation command executed successfully.",
            "stdout": "...".
            "stderr": "...".
            "return_code": 0
        }
    }
    ```
*   **Error (e.g., command failure):**
    ```json
    {
        "outputs": {
            "error": true,
            "message": "Command failed with exit code 1.",
            "stdout": "...".
            "stderr": "Error message...",
            "return_code": 1
        }
    }
    ```
