"""NIST NVD 2.0 API client.

Pulls CVE records and yields ``Document`` objects. The fetched JSON
is dense with metadata (CPEs, references, weaknesses); we flatten the
parts a security analyst actually wants to read into a single
``text`` blob and stash the rest under ``metadata``.

NVD's published rate limits (2024):
  * unauthenticated: 5 requests / 30 seconds
  * with API key:    50 requests / 30 seconds

We default to a polite 1.5s sleep between pages, which keeps us well
under the unauth limit even with a generous page size."""
from __future__ import annotations

import logging
import time
from typing import Iterable, Iterator

import httpx

from config import Settings
from ingest.document import Document, DocSource


_log = logging.getLogger("ingest.nvd")
_PAGE_SIZE = 100   # NVD default; max per-request without paging concern


def _flatten_cve(item: dict) -> Document | None:
    """Turn one NVD result item into a single text+metadata Document."""
    cve = item.get("cve") or {}
    cve_id = cve.get("id")
    if not cve_id:
        return None

    descs = cve.get("descriptions") or []
    en_desc = next((d.get("value", "") for d in descs
                     if d.get("lang") == "en"), "")

    metrics = cve.get("metrics") or {}
    cvss_v31 = (metrics.get("cvssMetricV31") or [{}])[0].get("cvssData", {})
    cvss_score = cvss_v31.get("baseScore")
    cvss_severity = cvss_v31.get("baseSeverity")

    weaknesses = []
    for w in cve.get("weaknesses") or []:
        for d in w.get("description") or []:
            v = d.get("value")
            if v and v != "NVD-CWE-Other":
                weaknesses.append(v)
    weaknesses = sorted(set(weaknesses))

    refs = [r.get("url") for r in (cve.get("references") or [])
            if isinstance(r.get("url"), str)][:8]

    parts = [
        f"# {cve_id}",
        f"Severity: {cvss_severity or 'unknown'}"
        + (f" (CVSS {cvss_score})" if cvss_score else ""),
    ]
    if weaknesses:
        parts.append("Weaknesses: " + ", ".join(weaknesses))
    parts.append("")
    parts.append(en_desc)
    if refs:
        parts.append("")
        parts.append("References:")
        parts.extend(f"  - {u}" for u in refs)

    return Document(
        text="\n".join(parts).strip(),
        source=DocSource.NVD,
        source_id=cve_id,
        metadata={
            "published":     cve.get("published"),
            "lastModified":  cve.get("lastModified"),
            "cvss_score":    cvss_score,
            "cvss_severity": cvss_severity,
            "weaknesses":    weaknesses,
            "ref_urls":      refs,
        },
    )


def fetch_recent_cves(
    settings: Settings,
    *,
    limit: int,
    page_size: int = _PAGE_SIZE,
    polite_sleep_s: float = 1.5,
    client: httpx.Client | None = None,
) -> Iterator[Document]:
    """Yield up to ``limit`` recent CVEs as Documents."""
    if limit <= 0:
        return
    headers = {"Accept": "application/json"}
    if settings.nvd_api_key:
        headers["apiKey"] = settings.nvd_api_key

    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    yielded = 0
    start_index = 0
    try:
        while yielded < limit:
            params = {
                "resultsPerPage": min(page_size, limit - yielded),
                "startIndex": start_index,
            }
            r = client.get(settings.nvd_api_base, headers=headers, params=params)
            if r.status_code >= 400:
                _log.warning("NVD HTTP %d, stopping ingest", r.status_code)
                return
            data = r.json()
            items = data.get("vulnerabilities") or []
            if not items:
                return
            for it in items:
                doc = _flatten_cve(it)
                if doc is None:
                    continue
                yield doc
                yielded += 1
                if yielded >= limit:
                    return
            start_index += len(items)
            if start_index >= (data.get("totalResults") or 0):
                return
            time.sleep(polite_sleep_s)
    finally:
        if owns_client:
            client.close()


def fetch_to_list(settings: Settings, *, limit: int) -> list[Document]:
    """Eager wrapper for callers that want a finite list."""
    return list(fetch_recent_cves(settings, limit=limit))


def fetch_cve_by_id(
    settings: Settings,
    cve_id: str,
    *,
    client: httpx.Client | None = None,
    polite_sleep_s: float = 1.5,
) -> Document | None:
    """Look up exactly one CVE by ID. Useful for seeding the corpus
    with named historical CVEs (Log4Shell, Heartbleed, etc.) that the
    eval set references."""
    headers = {"Accept": "application/json"}
    if settings.nvd_api_key:
        headers["apiKey"] = settings.nvd_api_key
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        r = client.get(settings.nvd_api_base, headers=headers,
                        params={"cveId": cve_id})
        if r.status_code >= 400:
            _log.warning("NVD by-id %s -> HTTP %d", cve_id, r.status_code)
            return None
        data = r.json()
        items = data.get("vulnerabilities") or []
        if not items:
            return None
        time.sleep(polite_sleep_s)
        return _flatten_cve(items[0])
    finally:
        if owns_client:
            client.close()


def fetch_cves_by_ids(
    settings: Settings,
    cve_ids: list[str],
) -> list[Document]:
    """Sequentially look up a list of CVE IDs, sleeping between
    requests to stay under the unauth rate limit."""
    out: list[Document] = []
    with httpx.Client(timeout=30.0) as client:
        for cid in cve_ids:
            doc = fetch_cve_by_id(settings, cid, client=client)
            if doc is not None:
                out.append(doc)
    return out
