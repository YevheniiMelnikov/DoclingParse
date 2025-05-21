import os, re, sys, json, base64, logging, subprocess, tomllib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from tempfile import NamedTemporaryFile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from contextlib import suppress

# ── CONFIG ────────────────────────────────────────────────────────────────────
CONFIG_FILE = Path("docling_config.toml")
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "rb") as f:
        raw_cfg = tomllib.load(f)
else:
    raw_cfg = {}

context_radius = int(raw_cfg.get("context_radius", 15))

# OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

MODEL_ID = "openai/gpt-4o"
TIMEOUT   = 120
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type":  "application/json"
}

DOCS_DIR = Path("docs")
OUT_DIR  = Path("parsed_output")
ENV = {**os.environ, "DOCLING_CONFIG": str(CONFIG_FILE.resolve())}

IMG_RE   = re.compile(r"!\[.*?]\((.*?)\)")
TABLE_RE = re.compile(r"^\|.*\|\s*$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ── PROMPTS ───────────────────────────────────────────────────────────────────
def prompt_image(ctx: str) -> str:
    return (
        "You are an expert technical writer.\n"
        "Write a precise, information-dense description of the image for RAG.\n"
        "• If it is a chart - mention axes, units, key figures & trends.\n"
        "• If diagram/screenshot - list main labelled elements.\n"
        "• Transcribe visible text verbatim. 3-6 sentences.\n\n"
        f"Context near image:\n{ctx}"
    )

def prompt_table(ctx: str) -> str:
    return (
        "You are an expert data analyst.\n"
        "Describe the table below for RAG.\n"
        "Explain every column, summarise rows, note patterns & outliers.\n"
        "Write 4-8 sentences. Do not reproduce the raw table.\n\n"
        f"Context near table:\n{ctx}"
    )

# ── UTILITIES ────────────────────────────────────────────────────────────────
def ctx_window(md: list[str], idx: int) -> str:
    return "".join(md[max(0, idx-context_radius): idx] +
                   md[idx+1: idx+1+context_radius]).strip()

def img_to_data_uri(path: Path) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"

def call_openrouter(img_path: Path, prompt: str, tag: str) -> str:
    logging.info("OpenRouter → %s", tag)
    img_data_uri = img_to_data_uri(img_path)
    body = {
        "model": MODEL_ID,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_data_uri}}
            ]
        }],
        "temperature": 0.2,
        "max_tokens": 300
    }
    try:
        t0 = perf_counter()
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=HEADERS, json=body, timeout=TIMEOUT
        )
        r.raise_for_status()
        elapsed = perf_counter() - t0
        logging.info("OpenRouter ✓ %.1fs (%s)", elapsed, tag)
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error("OpenRouter failed (%s): %s", tag, str(e))
        return "[ERROR] generation failed."

def render_table(table_md: str) -> Path:
    logging.info("Rendering table to image")
    rows = [l.strip() for l in table_md.strip().splitlines() if l.strip()]
    headers = rows[0].strip("|").split("|")
    data = [r.strip("|").split("|") for r in rows[2:]]
    df = pd.DataFrame(data, columns=[h.strip() for h in headers])
    fig, ax = plt.subplots(
        figsize=(min(12, len(headers)*1.8), min(0.7*len(data)+1, 14))
    )
    ax.axis("off")
    ax.table(cellText=df.values, colLabels=df.columns,
             cellLoc="center", loc="center")
    with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        fig.savefig(tmp.name, bbox_inches="tight", dpi=160)
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

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
def process_pdf(pdf: Path):
    base = pdf.stem
    artifacts = OUT_DIR / f"{base}_artifacts"
    md_path = OUT_DIR / f"{base}.md"

    docling_extract(pdf, artifacts)
    if not md_path.exists():
        logging.warning("No markdown produced for %s", pdf.name)
        return

    md_lines = md_path.read_text(encoding="utf-8").splitlines(keepends=True)
    result_lines, i, n = [], 0, len(md_lines)

    while i < n:
        line = md_lines[i]

        if IMG_RE.search(line):
            img_path = Path(IMG_RE.search(line).group(1))
            if not img_path.is_absolute():
                img_path = OUT_DIR / img_path
            desc = call_openrouter(
                img_path,
                prompt_image(ctx_window(md_lines, i)),
                f"img:{img_path.name}"
            )
            result_lines += [line, f"\n> **Image description:** {desc}\n"]
            i += 1
            continue

        if TABLE_RE.match(line):
            tbl_lines, start = [line], i
            while i+1 < n and TABLE_RE.match(md_lines[i+1]):
                i += 1
                tbl_lines.append(md_lines[i])
            img = render_table("".join(tbl_lines))
            desc = call_openrouter(
                img,
                prompt_table(ctx_window(md_lines, start)),
                f"tbl:{start}-{i}"
            )
            with suppress(FileNotFoundError):
                img.unlink()
            result_lines += [f"\n> **Table description:** {desc}\n"]
            i += 1
            continue

        result_lines.append(line)
        i += 1

    md_path.write_text("".join(result_lines), encoding="utf-8")
    logging.info("Finished %s → %s", pdf.name, md_path.name)

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
