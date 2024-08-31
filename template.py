import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Define the list of files and directories to create
list_of_files = [
    "data/dialog_talk_agent.xlsx",
    "data/test_data.csv",
    "chatbot/__init__.py",
    "chatbot/preprocess_query.py",
    "chatbot/drug_info.py",
    "chatbot/advanced_chatbot.py",
    "requirement.txt",
    ".gitignore"
    "setup.py",
    "main.py",
    "README.md"
]

def create_project_structure(file_list):
    for filepath in file_list:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)

        if filedir != "":
            # Create directories if they do not exist
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory: {filedir}")

        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            # Create empty files
            with open(filepath, 'w') as f:
                pass
            logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"{filename} already exists")

# Run the function to create the project structure
create_project_structure(list_of_files)
