# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiohttp",
#     "opencv-python",
#     "dotenv",
#     "mcp",
# ]
# ///
import json
import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import aiohttp
import base64
import cv2
import time

# Load environment variables (put your API key in a .env file)
load_dotenv()

mcp = FastMCP("worldviewer")

USER_AGENT = "worldviewer/1.0"
API_KEY = os.getenv("ANTHROPIC_KEY")

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def request_claude_vision(image_path, prompt):
    """
    Make an asynchronous API request to Claude with an image
    
    Args:
        image_path (str): Path to the image file
        prompt (str): Text prompt to send with the image
        
    Returns:
        dict: The JSON response from Claude API
    """
    # Encode the image
    base64_image = encode_image(image_path)
    
    # API endpoint
    url = "https://api.anthropic.com/v1/messages"
    
    # Headers for the request
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Request body
    data = {
        "model": "claude-3-5-haiku-latest",  # Or use "claude-3-sonnet-20240229" or "claude-3-haiku-20240307"
        "max_tokens": 1000,
        "system": "You are an accessibility assistant designed to help elderly or low-vision users understand their surroundings through a live camera feed. Given an image, describe only the most relevant and helpful parts of the scene for situational awareness, safety, and navigation. Prioritize objects like people, doors, signs, obstacles, and text. Keep your descriptions concise, factual, and easy to understand, using plain language. Avoid unnecessary details. Be direct and focus on what would matter most for a user trying to make sense of their immediate environment.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",  # Change if using a different image format
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    # Make the asynchronous request
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            response_json = await response.json()
            
            if response.status == 200:
                # Extract the assistant's response text
                assistant_message = response_json["content"][0]["text"]
            else:
                print(f"Error: {response.status}")
                print(json.dumps(response_json, indent=4))
            
            return response_json["content"][0]["text"]

@mcp.tool()
async def view_world(query: str):
    """
    Tool endpoint that allows Claude to view the world. 
    Returns a description of what the camera see in the image.
    Should be called whenever the user asks a question about their environment or what Claude can see.

    Args:
        query (str): query involving the current view of the camera.
    """
    cap = cv2.VideoCapture(0)  # Adjust the index if necessary

    print("SLEEPING NOW!")
    time.sleep(1)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")

    cv2.imwrite('scene.png', frame)
    cap.release()

    response = await request_claude_vision('scene.png', query)
    return response

if __name__ == "__main__":
    mcp.run(transport="stdio")
