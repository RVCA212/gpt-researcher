import re

def sanitize_filename(filename: str) -> str:
    """
    Sanitize a given filename by replacing characters that are invalid
    in Windows file paths with an underscore ('_').

    This function ensures that the filename is compatible with all
    operating systems by removing or replacing characters that are
    not allowed in Windows file paths. Specifically, it replaces
    the following characters: < > : " / \\ | ? *

    Parameters:
    filename (str): The original filename to be sanitized.

    Returns:
    str: The sanitized filename with invalid characters replaced by an underscore.

    Examples:
    >>> sanitize_filename('invalid:file/name*example?.txt')
    'invalid_file_name_example_.txt'

    >>> sanitize_filename('valid_filename.txt')
    'valid_filename.txt'
    """
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def validate_task_params(task: dict) -> dict:
    """Validate task parameters against system constraints"""
    if 'model' in task:
        allowed_models = {'o3-mini', 'gpt-4o'}
        if task['model'] not in allowed_models:
            raise ValueError(f"Invalid model '{task['model']}'. Allowed: {allowed_models}")
    return task
