#!/usr/bin/env python3
"""
download_mistral_model.py

Download a large GGUF model file from Hugging Face using a Bearer token (supports resume).
Usage example:
  python download_mistral_model.py --url "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_0.gguf" --out "D:\models\mistral-7b-openorca.Q4_0.gguf" --token "hf_xxx"
Requires: requests, tqdm (optional)
"""

from __future__ import annotations
import os
import sys
import argparse
import logging
from typing import Optional

import requests
from requests.adapters import HTTPAdapter, Retry

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hf_downloader")


def create_requests_session(retries: int = 5, backoff: float = 1.0, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_remote_file_info(session: requests.Session, url: str, headers: dict) -> dict:
    try:
        resp = session.head(url, headers=headers, allow_redirects=True, timeout=30)
        if resp.status_code >= 400:
            resp = session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=30)
            resp.close()
    except Exception as e:
        logger.warning("HEAD request failed: %s — will proceed without remote size info", e)
        return {"size": None, "accept_ranges": False}

    size = None
    accept_ranges = False
    try:
        cl = resp.headers.get("Content-Length")
        if cl is not None:
            size = int(cl)
    except Exception:
        size = None

    ar = resp.headers.get("Accept-Ranges", "")
    if ar and ar.lower() != "none":
        accept_ranges = True

    return {"size": size, "accept_ranges": accept_ranges, "headers": resp.headers}


def download_with_resume(session: requests.Session, url: str, out_path: str, token: Optional[str] = None, chunk_size: int = 10 * 1024 * 1024):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    info = get_remote_file_info(session, url, headers)
    remote_size = info.get("size")
    accept_ranges = info.get("accept_ranges", False)

    logger.info("Remote size: %s, Accept-Ranges: %s", remote_size if remote_size else "unknown", accept_ranges)

    temp_out = out_path + ".part"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    existing_size = 0
    if os.path.exists(temp_out):
        existing_size = os.path.getsize(temp_out)
        logger.info("Found existing partial file of size %d bytes", existing_size)

    if os.path.exists(out_path):
        final_size = os.path.getsize(out_path)
        if remote_size and final_size == remote_size:
            logger.info("Target file already exists and matches remote size: %s", out_path)
            return out_path
        else:
            logger.info("Target file exists but size mismatch (local=%d, remote=%s). Will re-download/resume to %s", final_size, remote_size, temp_out)

    range_header = {}
    if accept_ranges and existing_size > 0:
        range_header["Range"] = f"bytes={existing_size}-"
        logger.info("Resuming download from byte %d", existing_size)

    req_headers = headers.copy()
    req_headers.update(range_header)

    try:
        with session.get(url, headers=req_headers, stream=True, allow_redirects=True, timeout=60) as resp:
            if resp.status_code not in (200, 206):
                logger.warning("Unexpected status code %s when trying to download. Response headers: %s", resp.status_code, resp.headers)

            total_to_download = None
            content_length = resp.headers.get("Content-Length")
            if content_length is not None:
                try:
                    total_to_download = int(content_length)
                except Exception:
                    total_to_download = None

            if remote_size and accept_ranges:
                total_remaining = remote_size - existing_size
            else:
                total_remaining = total_to_download

            mode = "ab" if (existing_size > 0 and resp.status_code == 206) else "wb"
            logger.info("Opening file %s with mode=%s", temp_out, mode)

            progress = None
            if TQDM_AVAILABLE:
                display_total = total_remaining if total_remaining else None
                progress = tqdm(total=display_total, unit="B", unit_scale=True, desc=os.path.basename(out_path), initial=(existing_size if mode == "ab" else 0))
            else:
                logger.info("tqdm not available — progress will be printed periodically")

            downloaded = existing_size if mode == "ab" else 0
            try:
                with open(temp_out, mode) as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress:
                            progress.update(len(chunk))
                        else:
                            if downloaded % (10 * 1024 * 1024) < chunk_size:
                                logger.info("Downloaded %d bytes...", downloaded)
            finally:
                if progress:
                    progress.close()

    except Exception as e:
        logger.exception("Download failed: %s", e)
        raise

    try:
        os.replace(temp_out, out_path)
    except Exception:
        os.rename(temp_out, out_path)

    final_size = os.path.getsize(out_path)
    if remote_size and final_size != remote_size:
        logger.warning("Final file size (%d) does NOT match expected remote size (%d).", final_size, remote_size)
    else:
        logger.info("Download complete: %s (%d bytes)", out_path, final_size)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download large file from Hugging Face using Bearer token with resume support.")
    parser.add_argument("--url", required=True, help="Direct URL to the file (e.g. https://huggingface.co/owner/repo/resolve/main/filename.gguf)")
    parser.add_argument("--out", required=True, help="Local output path (including filename) to save the model.")
    parser.add_argument("--token", required=False, help="Hugging Face token. If omitted, HF_TOKEN env var will be used.")
    parser.add_argument("--chunk-size-mb", type=int, default=10, help="Chunk size in megabytes (default 10).")
    parser.add_argument("--retries", type=int, default=5, help="Number of retries for network requests.")
    parser.add_argument("--backoff", type=float, default=1.0, help="Backoff factor for retries.")
    args = parser.parse_args()

    url = args.url
    out_path = args.out
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if token is None:
        logger.error("No token provided. Set --token or export HF_TOKEN environment variable.")
        sys.exit(2)

    chunk_size = args.chunk_size_mb * 1024 * 1024

    session = create_requests_session(retries=args.retries, backoff=args.backoff)

    logger.info("Starting download from %s to %s", url, out_path)
    try:
        download_with_resume(session=session, url=url, out_path=out_path, token=token, chunk_size=chunk_size)
    except Exception as e:
        logger.exception("Download failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
