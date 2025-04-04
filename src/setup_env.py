# temp/src/setup_env.py
import os
from dotenv import load_dotenv, set_key

# Define the path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') # Assumes .env is in the parent (temp) folder

def setup_groq_key():
    """Prompts the user for their Groq API key and saves it to the .env file."""
    load_dotenv(dotenv_path=dotenv_path) # Load existing vars if any

    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        print(f"Groq API key found in {dotenv_path}.")
        overwrite = input("Do you want to overwrite it? (y/n): ").lower().strip()
        if overwrite != 'y':
            print("Using existing key.")
            return

    new_api_key = input("Please enter your Groq API key: ").strip()

    if new_api_key:
        # Use set_key which creates the file if it doesn't exist
        set_key(dotenv_path=dotenv_path, key_to_set="GROQ_API_KEY", value_to_set=new_api_key)
        print(f"Groq API key saved to {dotenv_path}")

        # Verify it was set in the current environment for immediate use (optional)
        os.environ["GROQ_API_KEY"] = new_api_key
        print("API Key also set for the current session.")
    else:
        print("No API key entered.")

if __name__ == "__main__":
    setup_groq_key()