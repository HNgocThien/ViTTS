import os

def convert_to_lf(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    with open(filepath, 'rb') as f:
        content = f.read()
    
    # Replace CRLF with LF
    new_content = content.replace(b'\r\n', b'\n')
    
    if new_content != content:
        with open(filepath, 'wb') as f:
            f.write(new_content)
        print(f"Converted {filepath} to LF.")
    else:
        print(f"{filepath} is already LF or has no CRLF.")

files_to_fix = [
    r".\mimic-recording-studio\backend\start_prod.sh",
    r".\mimic-recording-studio\backend\gunicorn_conf.py",
    r".\mimic-recording-studio\frontend\start.sh"
]

for f in files_to_fix:
    convert_to_lf(f)
