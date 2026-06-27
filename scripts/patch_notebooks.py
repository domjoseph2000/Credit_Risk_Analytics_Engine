"""
patch_notebooks.py
==================
Rewrites the hard-coded Windows PROJECT_ROOT path in each notebook to the
container path /data, and updates os.chdir() accordingly.

Run once at container startup (see Dockerfile ENTRYPOINT / CMD).
"""
import json
import os
import re
import sys

NB_DIR          = os.environ.get("HOMECREDIT_NB_DIR", "/app/notebooks")
CONTAINER_ROOT  = os.environ.get("HOMECREDIT_DATA_ROOT", "/data")
WINDOWS_ROOT_RE = re.compile(r'r"[^"]*\\\\[^"]*Home Credit[^"]*"')

def patch_notebook(path: str) -> None:
    with open(path) as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        new_source = []
        for line in cell["source"]:
            # Replace Windows hard-coded path
            new_line = WINDOWS_ROOT_RE.sub(f'"{CONTAINER_ROOT}"', line)
            # Replace backslash path separators in parquet/csv string literals
         #   new_line = new_line.replace("src\\\\intermediate", "src/intermediate")
            new_line = new_line.replace("src\\\\Variable-Selection", "src/Variable-Selection")
            new_line = new_line.replace("data\\\\Input", "data/Input")
            if new_line != line:
                changed = True
            new_source.append(new_line)
        cell["source"] = new_source

    if changed:
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"  Patched: {os.path.basename(path)}")
    else:
        print(f"  OK (no changes): {os.path.basename(path)}")


if __name__ == "__main__":
    print(f"Patching notebooks in {NB_DIR} → root = {CONTAINER_ROOT}")
    nb_files = [f for f in os.listdir(NB_DIR) if f.endswith(".ipynb")]
    if not nb_files:
        print("No notebooks found!", file=sys.stderr)
        sys.exit(1)
    for nb_file in sorted(nb_files):
        patch_notebook(os.path.join(NB_DIR, nb_file))
    print("Done.")
