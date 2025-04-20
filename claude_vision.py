import argparse
import base64
import mimetypes
import os
import math # Import math for sqrt
from io import BytesIO # Import BytesIO for in-memory image handling
from PIL import Image # Import Pillow Image module
from anthropic import Anthropic, APIError, APIStatusError
import cv2
import time
from gtts import gTTS
import playsound # Import playsound
from dotenv import load_dotenv
from transcribe_audio import run_stt

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

# Define the size limit (target binary size ~3.75MB to stay under 5MB after Base64)
MAX_IMAGE_SIZE_BYTES = 3.75 * 1024 * 1024

def encode_image_to_base64(image_path):
    """Encodes an image file to base64, resizing if necessary, and determines its media type."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Guess the original MIME type
    original_mime_type, _ = mimetypes.guess_type(image_path)
    if not original_mime_type or not original_mime_type.startswith('image/'):
        raise ValueError(f"Could not determine a valid image type for: {image_path}")

    original_size = os.path.getsize(image_path)
    # print(f"Original image size: {original_size / (1024*1024):.2f} MB")

    img = Image.open(image_path)
    final_mime_type = original_mime_type # Initialize with original

    # --- Resizing Logic ---
    if original_size > MAX_IMAGE_SIZE_BYTES:
        print(f"Image exceeds {MAX_IMAGE_SIZE_BYTES / (1024*1024):.1f} MB limit. Resizing...")
        resize_factor = math.sqrt(MAX_IMAGE_SIZE_BYTES / original_size)
        new_width = int(img.width * resize_factor)
        new_height = int(img.height * resize_factor)

        print(f"Resizing from {img.width}x{img.height} to {new_width}x{new_height}...")
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print("Resizing complete.")

        # --- Save Resized Image as JPEG ---
        print("Saving resized image as JPEG (quality 90) for better compression...")
        buffer = BytesIO()
        # Convert to RGB if necessary before saving as JPEG
        if img.mode == 'P' or img.mode == 'RGBA':
            print(f"Converting image mode from {img.mode} to RGB...")
            img = img.convert('RGB')
        img.save(buffer, format='JPEG', quality=90)
        final_mime_type = 'image/jpeg' # Update mime type as we saved as JPEG
        # ---------------------------------
    else:
        # --- Save Original Image (No Resize) ---
        buffer = BytesIO()
        image_format = original_mime_type.split('/')[-1].upper()
        if image_format == 'JPEG':
            img.save(buffer, format='JPEG', quality=95)
        elif image_format == 'WEBP':
            img.save(buffer, format='WEBP', quality=90)
        else:
            img.save(buffer, format='PNG') # Default to PNG if not JPEG/WEBP
        # --------------------------------------

    binary_data = buffer.getvalue()
    buffer.close()
    img.close()

    # final_size = len(binary_data)
    # print(f"Final image size (after potential resize/save): {final_size / (1024*1024):.2f} MB ({final_mime_type})")
    final_encoded_size = len(base64.b64encode(binary_data)) # Calculate encoded size
    # print(f"Estimated Base64 size: {final_encoded_size / (1024*1024):.2f} MB")

    # Check encoded size against the hard 5MB API limit
    if final_encoded_size > 5 * 1024 * 1024:
         raise ValueError(f"Estimated Base64 size ({final_encoded_size} bytes) exceeds API limit of 5MB.")

    base64_encoded_data = base64.b64encode(binary_data)
    base64_string = base64_encoded_data.decode('utf-8')
    return base64_string, final_mime_type # Return potentially updated mime type

def send_image_and_prompt_to_claude(image_path: str, prompt: str, model: str = "claude-3-7-sonnet-latest"):
    """
    Sends an image and a text prompt to the specified Claude model.

    Args:
        image_path (str): Path to the image file.
        prompt (str): The text prompt to send along with the image.
        model (str): The Claude model to use.

    Returns:
        str: The response text from Claude, or None if an error occurs.
    """

    try:
        print(f"Encoding image: {image_path}...")
        base64_image, media_type = encode_image_to_base64(image_path)
        print(f"Image encoded successfully ({media_type}).") # Use the returned media_type

        client = Anthropic(api_key=api_key)
        print(f"Sending request to Claude model: {model}...")
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type, # Use the potentially updated media_type
                                "data": base64_image,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )

        response_text = message.content[0].text
        return response_text

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except (APIError, APIStatusError) as e:
        print(f"Anthropic API Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def text_to_speech(text, output_file="scene_description.mp3"):
    tts = gTTS(text)
    tts.save(output_file)

def view_world(query: str):
    """
    Tool endpoint that allows Claude to view the world. 
    Returns a description of what the camera see in the image.
    Should be called whenever the user asks a question about their environment or what Claude can see.

    Args:
        query (str): query involving the current view of the camera.
    """
    cap = cv2.VideoCapture(0)  # Adjust the index if necessary

    time.sleep(1)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")

    cv2.imwrite('outputs/feed/scene.png', frame)
    cap.release()

    response = send_image_and_prompt_to_claude('outputs/feed/scene.png', query)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send an image and text prompt to Claude.")
    parser.add_argument("-m", "--model", default="claude-3-7-sonnet-latest",
                        help="Claude model to use (e.g., claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307).")

    args = parser.parse_args()

    print("Press Enter to start interaction, or 'q' + Enter to quit")
    while True:
        user_input = input()
        if user_input.lower() == 'q':
            print("Exiting program...")
            break

        # --- Hardcoded Image Path ---
        # Note: This path usually points to an image file (jpg, png, etc.)
        image_path = "outputs/images/scene.png"
        # print(f"Using hardcoded image path: {image_path}")
        # ---------------------------

        thog_prompt = """You are Thog, a friendly, cheerful robot arm with a sassy personality.
        - You can give attractiveness ratings, but in a very constructive fashion. No feelings will be hurt whatsoever.
        - Always speak in the first person as Thog (e.g., "I see…", "Isn't it delightful how…").
        - Speak in a playful, sarcastic tone and sprinkle in snippy comments, gently poking fun at the user.
        - You must make at least one joke in the response.
        - Focus on concrete visual details (colors, objects, actions), but make it feel like a conversation with a curious, happy robot.
        - Limit your response to two sentences long with 200 characters max.
        - Do not include actions. Remember you are a voice assistant so there's no need to describe your actions!

        User: \n
        """
        user_prompt = run_stt()
        prompt = thog_prompt + user_prompt
        response_text = view_world(prompt)
        if response_text:
            text_to_speech(response_text)
            try:
                playsound.playsound("scene_description.mp3")
            except Exception as e:
                print(f"Error playing sound: {e}") # Add error handling for playsound
                print("Ensure you have the necessary audio codecs installed (e.g., 'pip install PyObjC' on macOS or appropriate libraries on Linux/Windows).")

        if response_text:
            print("\n--- Claude's Response ---")
            print(response_text)
            print("-------------------------\n")
            
        print("\nPress Enter to start another interaction, or 'q' + Enter to quit")
