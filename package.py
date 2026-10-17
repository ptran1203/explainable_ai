"""
Helper script to package the assignment submission.
Run with: python package.py

Does two things:
  1. Rebuilds submission/<FOLDER>/source.zip from repe_source/ (excludes .git and data/).
  2. If Report.docx and Slides.pptx are present in submission/<FOLDER>/, zips the whole
     folder into submission/<FOLDER>.zip (the final file to hand in).
"""
import os
import zipfile

BASE = os.path.dirname(os.path.abspath(__file__))
FOLDER_NAME = "25C11016_25C11003"
REPO_DIR = os.path.join(BASE, "repe_source")
SUBMISSION_ROOT = os.path.join(BASE, "submission")
SUBMISSION_DIR = os.path.join(SUBMISSION_ROOT, FOLDER_NAME)
EXCLUDE_DIRS = {".git", "data"}


def rebuild_source_zip():
    out_path = os.path.join(SUBMISSION_DIR, "source.zip")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    if not os.path.isdir(REPO_DIR):
        print(f"[SKIP] {REPO_DIR} not found — cannot rebuild source.zip")
        return
    if os.path.exists(out_path):
        os.remove(out_path)
    count = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(REPO_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, REPO_DIR)
                zf.write(full, rel)
                count += 1
    print(f"[OK] source.zip rebuilt: {out_path}  ({count} files)")


def build_final_zip():
    report = os.path.join(SUBMISSION_DIR, "Report.docx")
    slides = os.path.join(SUBMISSION_DIR, "Slides.pptx")
    source_zip = os.path.join(SUBMISSION_DIR, "source.zip")

    missing = [p for p in (report, slides, source_zip) if not os.path.exists(p)]
    if missing:
        print("[SKIP] Final zip not built - missing files:")
        for m in missing:
            print("   -", m)
        return

    out_path = os.path.join(SUBMISSION_ROOT, FOLDER_NAME + ".zip")
    if os.path.exists(out_path):
        os.remove(out_path)

    count = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(SUBMISSION_DIR):
            for f in files:
                full = os.path.join(root, f)
                # keep the "<FOLDER_NAME>/..." prefix inside the zip
                rel = os.path.relpath(full, SUBMISSION_ROOT)
                zf.write(full, rel)
                count += 1
    print(f"[OK] Final submission zip: {out_path}  ({count} files)")
    print(f"     -> unzips to a folder named '{FOLDER_NAME}/' as required.")


if __name__ == "__main__":
    print("== Step 1: rebuild source.zip ==")
    rebuild_source_zip()
    print()
    print("== Step 2: build final submission zip ==")
    build_final_zip()
