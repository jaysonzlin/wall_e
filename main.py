from fastapi import FastAPI, HTTPException, Body
import subprocess
import shlex
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow cross-origin requests from Claude Desktop
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you might want to restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MCP endpoint to declare server capabilities
@app.get("/mcp")
async def mcp_info():
    """
    MCP endpoint that declares this server's capabilities to Claude.
    """
    return {
        "status": "ok",
        "capabilities": ["eval_service"]
    }

# MCP endpoint to handle evaluation requests
@app.post("/mcp/eval_service")
async def mcp_eval_service(payload: dict = Body(...)):
    """
    MCP endpoint that handles evaluation requests from Claude.
    Expected payload format: {"parameters": {"output_dir": "/path/to/your/output/dir"}}
    """
    # Extract parameters from the MCP payload structure
    if not payload.get("parameters") or not payload["parameters"].get("output_dir"):
        raise HTTPException(status_code=400, detail="'output_dir' must be provided in the parameters.")
    
    output_dir = payload["parameters"]["output_dir"]
    
    # Run the evaluation
    eval_result = await run_evaluation(output_dir)
    
    # Return result in MCP format
    return {
        "status": "ok",
        "result": eval_result
    }

# Original evaluation function, modified to accept output_dir directly
async def run_evaluation(output_dir: str):
    """
    Activates the 'lerobot' conda environment and runs the evaluation script.
    """
    if not output_dir:
        raise HTTPException(status_code=400, detail="'output_dir' must be provided.")

    # Construct the command to run the evaluation script within the conda environment
    eval_script_path = os.path.expanduser("~/lerobot/lerobot/scripts/eval.py")
    policy_path = os.path.join(output_dir, "checkpoints", "last", "pretrained_model")
    
    command = f"conda run -n lerobot python {eval_script_path} --policy.path={policy_path}"

    print(f"Executing command: {command}")

    try:
        # Using subprocess.run to execute the command
        result = subprocess.run(shlex.split(command), capture_output=True, text=True, check=True, shell=False)
        
        print("Command executed successfully:")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        
        return {
            "message": "Evaluation command executed successfully.",
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except FileNotFoundError:
        print(f"Error: 'conda' command not found. Make sure Conda is installed and in the system's PATH.")
        raise HTTPException(status_code=500, detail="'conda' command not found. Ensure Conda is installed and accessible.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise HTTPException(status_code=500, 
                            detail=f"Command failed with exit code {e.returncode}. Error: {e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Keep the original endpoint for backward compatibility
@app.post("/run_eval")
async def original_run_evaluation(payload: dict = Body(...)):
    """
    Original endpoint for backward compatibility.
    Example request body: {"output_dir": "/path/to/your/output/dir"}
    """
    output_dir = payload.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=400, detail="'output_dir' must be provided in the request body.")
    
    return await run_evaluation(output_dir)

if __name__ == "__main__":
    # Run the server - change to 0.0.0.0 to make it accessible from other machines
    uvicorn.run(app, host="0.0.0.0", port=8000)
