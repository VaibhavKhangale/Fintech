"""Print the text outputs of an executed notebook, for a quick review."""
import sys

import nbformat


def main() -> None:
    notebook = nbformat.read(sys.argv[1], as_version=4)
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                print("ERROR", index, output.ename, output.evalue)
            text = output.get("text") or output.get("data", {}).get("text/plain", "")
            if text and not str(text).startswith(("<Figure", "<pandas")):
                print(f"-- cell {index}\n{str(text)[:limit]}")


if __name__ == "__main__":
    main()
