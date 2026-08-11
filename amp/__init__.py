from pathlib import Path

def get_project_root():
    """ Get project root path"""
    return str(Path(__file__).resolve().parent.parent)