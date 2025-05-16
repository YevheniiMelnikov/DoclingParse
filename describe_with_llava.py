import os
import re
import sys
import logging
import subprocess
import tomllib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tempfile import NamedTemporaryFile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from contextlib import suppress
from typing import Optional

# --- Configuration ---
CONFIG_FILE = Path("docling_config.toml")
with open(CONFIG_FILE, "rb") as f:
    raw_cfg = tomllib.load(f)

@dataclass(frozen=True)
class VLM:
    path: Path
    model: Path
    mmproj: Path
    chat_template: str = "vicuna"
    ctx_size: int = 4096
    n_predict: int = 256
    timeout: int = 420

vlm = VLM(**raw_cfg["vlm"])
context_radius = int(raw_cfg.get("context_radius", 15))

DOCS_DIR = Path("docs")
OUT_DIR = Path("parsed_output")
ENV = {**os.environ, "DOCLING_CONFIG": str(CONFIG_FILE.resolve())}

IMG_RE = re.compile(r"!\[.*?]\((.*?)\)")
TABLE_RE = re.compile(r"^\|.*\|\s*$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# --- Prompt Templates ---
def prompt_image(ctx: str) -> str:
    return (
        "You are an expert technical writer. "
        "Generate a precise, information-dense description of the image for retrieval-augmented generation (RAG). "
        "If it is a chart, include axis titles, units, key figures, and trends. "
        "If it is a diagram or screenshot, list main labeled elements and relationships. "
        "Convert all visible text verbatim. Avoid vague words. Write 3–6 clear sentences.\n\n"
        f"Context (captions etc.):\n{ctx}"
    )

def prompt_table(ctx: str) -> str:
    return (
        "You are an expert data analyst. "
        "Describe the following table for retrieval-augmented generation. "
        "Explain every column, summarize rows, and identify patterns, outliers, and units. "
        "Do not include the original markdown table. Write 4–8 clear sentences.\n\n"
        f"Context (captions etc.):\n{ctx}"
    )

# --- Utilities ---
def ctx_window(md: list[str], idx: int) -> str:
    return "".join(md[max(0, idx - context_radius): idx] + md[idx + 1: idx + context_radius + 1]).strip()

def run_llava(img: Path, prompt: str, tag: str) -> str:
    logging.info("Running LLaVA on %s", tag)
    t0 = perf_counter()
    try:
        proc = subprocess.run([
            str(vlm.path), "-m", str(vlm.model),
            "--mmproj", str(vlm.mmproj),
            "--image", str(img),
            "--prompt", prompt,
            "--chat-template", vlm.chat_template,
            "--n-predict", str(vlm.n_predict),
            "--ctx-size", str(vlm.ctx_size),
        ], capture_output=True, text=True, env=ENV, timeout=vlm.timeout, check=True)
    except subprocess.CalledProcessError as e:
        logging.error("LLaVA failed (%s): %s", tag, (e.stderr or e.stdout).strip())
        return "[ERROR] generation failed."
    except subprocess.TimeoutExpired:
        logging.error("LLaVA timeout (%s)", tag)
        return "[ERROR] generation timeout."

    elapsed = perf_counter() - t0
    logging.info("LLaVA done (%s) in %.2fs", tag, elapsed)

    skip = ("llama_", "clip_", "build:", "load_")
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip() and not l.startswith(skip)]
    return lines[-1] if lines else "[WARN] empty response."

def render_table(table_md: str) -> Path:
    logging.info("Rendering table to image")
    rows = [l.strip() for l in table_md.strip().splitlines() if l.strip()]
    headers = rows[0].strip("|").split("|")
    data = [r.strip("|").split("|") for r in rows[2:]]
    df = pd.DataFrame(data, columns=[h.strip() for h in headers])
    fig, ax = plt.subplots(figsize=(min(12, len(headers) * 1.8), min(0.7 * len(data), 12)))
    ax.axis("off")
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig.savefig(tmp.name, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return Path(tmp.name)

def docling_extract(pdf: Path, out_artifacts: Path):
    if out_artifacts.exists():
        logging.info("Docling artifacts already exist for %s", pdf.name)
        return
    logging.info("Running Docling on %s", pdf.name)
    subprocess.run([
        "docling", str(pdf), "--pipeline", "standard",
        "--image-export-mode", "referenced",
        "--output", str(OUT_DIR), "--no-ocr"
    ], env=ENV, check=True)

# --- Per-file processing ---
def process_pdf(pdf: Path):
    base = pdf.stem
    artifacts = OUT_DIR / f"{base}_artifacts"
    md_src = OUT_DIR / f"{base}.md"
    md_dst = OUT_DIR / f"{base}.md"

    docling_extract(pdf, artifacts)
    if not md_src.exists():
        logging.warning("No markdown found for %s", pdf.name)
        return

    logging.info("Processing %s", pdf.name)
    md = md_src.read_text(encoding="utf-8").splitlines(keepends=True)
    out, i, n = [], 0, len(md)

    while i < n:
        line = md[i]

        if IMG_RE.search(line):
            img_path = Path(IMG_RE.search(line).group(1))
            if not img_path.is_absolute():
                img_path = OUT_DIR / img_path
            ctx = ctx_window(md, i)
            desc = run_llava(img_path, prompt_image(ctx), f"img:{img_path.name}")
            out += [line, f"\n> **Image description:** {desc}\n"]
            i += 1
            continue

        if TABLE_RE.match(line):
            tbl_lines, start = [line], i
            while i + 1 < n and TABLE_RE.match(md[i + 1]):
                i += 1
                tbl_lines.append(md[i])
            ctx = ctx_window(md, start)
            img = render_table("".join(tbl_lines))
            desc = run_llava(img, prompt_table(ctx), f"tbl:{start}-{i}")
            with suppress(FileNotFoundError):
                img.unlink()
            out += [f"\n> **Table description:** {desc}\n"]
            i += 1
            continue

        out.append(line)
        i += 1

    md_dst.parent.mkdir(parents=True, exist_ok=True)
    md_dst.write_text("".join(out), encoding="utf-8")
    logging.info("Finished %s → %s", pdf.name, md_dst.name)

# --- Entry Point ---
def main():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        logging.warning("No PDFs found in %s", DOCS_DIR)
        return
    for pdf in pdfs:
        process_pdf(pdf)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
