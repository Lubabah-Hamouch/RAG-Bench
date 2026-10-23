"""Download the corpus PDFs that are not redistributed with this repo.

    python scripts/fetch_corpus.py [--dest DIR] [--browser PATH]

Reads data/corpus/manifest.yaml and, for every entry with `local: false` whose file is missing,
fetches it into data/corpus/ (or --dest):

- arXiv papers are downloaded from https://arxiv.org/pdf/<id>v<n>, the version pinned in `file`.
- Web pages (Pinecone, Weaviate, LangChain docs) are not PDFs at their source, so they are printed
  to PDF with a headless Chrome or Edge, which the benchmark's copies were made with. These pages
  change over time, so a fresh copy can differ from the 2025-12-07 snapshot the recorded results
  used. Pass --browser if your Chrome/Edge is not found automatically.

Existing files are skipped. Exits non-zero if anything failed.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ragbench.ingest.manifest import DocumentSpec, load_manifest

ROOT = Path(__file__).resolve().parents[1]
ARXIV_ABS = re.compile(r"^https?://arxiv\.org/abs/(?P<id>\d{4}\.\d{4,5})$")
BROWSER_CANDIDATES = [
    "chrome",
    "google-chrome",
    "chromium",
    "msedge",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]
USER_AGENT = "ragbench-fetch-corpus/1.0 (+https://github.com/Lubabah-Hamouch/RAG-Bench)"


def find_browser(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() or shutil.which(explicit) else None
    for candidate in BROWSER_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
        if Path(candidate).exists():
            return candidate
    return None


def fetch_arxiv(spec: DocumentSpec, dest: Path) -> None:
    match = ARXIV_ABS.match(spec.source_url)
    version = re.search(r"v(\d+)\.pdf$", spec.file)
    if not match or not version:
        raise ValueError(f"cannot derive an arXiv PDF URL from {spec.source_url!r} / {spec.file!r}")
    url = f"https://arxiv.org/pdf/{match['id']}v{version[1]}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url} returned HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} is unreachable: {exc.reason}") from exc
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"{url} did not return a PDF")
    dest.write_bytes(data)


def fetch_webpage(spec: DocumentSpec, dest: Path, browser: str | None) -> None:
    if browser is None:
        raise RuntimeError(
            "no Chrome/Edge found to print the page to PDF; pass --browser PATH, or save "
            f"{spec.source_url} as a PDF named {spec.file!r} in {dest.parent} yourself"
        )
    # Reachability first, so a dead page is reported as such instead of printing an error page.
    request = urllib.request.Request(spec.source_url, headers={"User-Agent": USER_AGENT})
    try:
        urllib.request.urlopen(request, timeout=30).close()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{spec.source_url} returned HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{spec.source_url} is unreachable: {exc.reason}") from exc
    try:
        result = subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--virtual-time-budget=8000",
                f"--print-to-pdf={dest}",
                spec.source_url,
            ],
            capture_output=True,
            timeout=90,
            check=False,
        )
        stderr = result.stderr.decode(errors="replace").strip()[-300:]
    except subprocess.TimeoutExpired:
        stderr = "browser timed out"
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"browser did not produce a PDF for {spec.source_url}: {stderr}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=Path, default=ROOT / "data" / "corpus")
    parser.add_argument("--browser", help="Chrome/Edge executable used for web pages")
    args = parser.parse_args()

    manifest = load_manifest(ROOT / "data" / "corpus" / "manifest.yaml")
    args.dest.mkdir(parents=True, exist_ok=True)
    browser = find_browser(args.browser)

    downloaded: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    for spec in manifest.documents:
        target = args.dest / spec.file
        if spec.local or target.exists():
            skipped.append(spec.file)
            continue
        print(f"fetching {spec.file} <- {spec.source_url}")
        try:
            if ARXIV_ABS.match(spec.source_url):
                fetch_arxiv(spec, target)
            else:
                fetch_webpage(spec, target, browser)
            downloaded.append(spec.file)
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            target.unlink(missing_ok=True)
            failed.append((spec.file, str(exc)))

    print(f"\ndownloaded {len(downloaded)}, skipped {len(skipped)}, failed {len(failed)}")
    for name in downloaded:
        print(f"  downloaded: {name}")
    for name in skipped:
        print(f"  skipped (already present or ships in the repo): {name}")
    for name, reason in failed:
        print(f"  FAILED: {name}: {reason}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
