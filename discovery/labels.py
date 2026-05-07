"""Human-readable labels for SAE latents via the Neuronpedia HTTP API.

- **API**: direct ``requests`` GET to ``/api/feature/...`` (same contract as neuronpedia.org);
  parses current JSON including explanations shaped as ``{description, typeName, …}``.
  Uses small local exception types aligned with the ``neuronpedia`` client (401 / 429 / missing key).
  Keys via ``NEURONPEDIA_API_KEY`` or ``api_key=`` (never written to disk or logged here).
- **Cache**: JSON on disk under ``.cache/neuronpedia_labels/`` to avoid redundant calls.
- **Fallbacks**: missing keys, 404, rate limits, empty explanations, near-dead features.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import requests


class NPKeyMissingError(Exception):
    """No API key provided."""

    pass


class NPUnauthorizedError(Exception):
    """Unauthorized — invalid or missing credentials for the Neuronpedia API."""

    pass


class NPRateLimitError(Exception):
    """Neuronpedia API rate limit exceeded."""

    pass


LabelStatus = Literal[
    "ok",
    "cached",
    "no_explanation",
    "not_found",
    "dead_feature",
    "rate_limited",
    "unauthorized",
    "no_api_key",
    "error",
]


NO_EXPLANATION_TEXT = "(no explanation found)"
NOT_FOUND_TEXT = "(feature not found on Neuronpedia)"
DEAD_FEATURE_TEXT = "(dead or inactive feature — near-zero density)"
GENERIC_ERROR_TEXT = "(label fetch failed)"


@dataclass(frozen=True)
class ExplanationSnippet:
    """One explanation line from Neuronpedia."""

    text: str
    method: Optional[str] = None
    explainer_model: Optional[str] = None


@dataclass(frozen=True)
class FeatureLabel:
    """Resolved label payload for one latent (API + fallbacks)."""

    model_id: str
    source: str
    index: int
    primary_text: str
    explanations: tuple[ExplanationSnippet, ...]
    density: Optional[float]
    status: LabelStatus
    detail: Optional[str] = None
    neuronpedia_url: Optional[str] = None


def _default_cache_dir() -> Path:
    return Path(".cache") / "neuronpedia_labels"


def _cache_key(model_id: str, source: str, index: Union[int, str]) -> str:
    raw = f"{model_id}\n{source}\n{str(index).strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _sae_cache_dir(cache_root: Path, model_id: str, source: str) -> Path:
    """Directory keying labels by (model_id, source) = one SAE / feature set."""

    # Keep path-friendly but deterministic.
    mid = str(model_id).strip().replace("/", "_")
    src = str(source).strip().replace("/", "_")
    return cache_root / "by_sae" / mid / src


def _sae_index_cache_path(cache_root: Path, model_id: str, source: str) -> Path:
    """Single JSON map for this SAE: {index(str): payload}."""

    return _sae_cache_dir(cache_root, model_id, source) / "index_map.v1.json"


def _feature_url(model_id: str, source: str, index: int) -> str:
    return f"https://neuronpedia.org/{model_id}/{source}/{index}"


def _neuronpedia_api_base() -> str:
    use_local = os.getenv("USE_LOCALHOST", "false").lower() == "true"
    return "http://localhost:3000/api" if use_local else "https://neuronpedia.org/api"


def _fetch_feature_document(model_id: str, source: str, index: int, api_key: str) -> dict[str, Any]:
    """GET /feature/{model}/{layer}/{index} — tolerant of newer JSON than ``neuronpedia`` parses."""
    url = f"{_neuronpedia_api_base()}/feature/{model_id}/{source}/{index}"
    resp = requests.get(
        url,
        headers={"X-Api-Key": api_key, "Accept-Encoding": "gzip"},
        timeout=120,
    )
    if resp.status_code == 401:
        raise NPUnauthorizedError(
            "Unauthorized. Check NEURONPEDIA_API_KEY or pass api_key= to fetch_feature_label."
        )
    if resp.status_code == 429:
        raise NPRateLimitError(
            "Rate limit exceeded. Try later or reduce request rate."
        )
    resp.raise_for_status()
    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        raise NPInvalidPayloadError(str(e)) from e
    if not isinstance(data, dict):
        raise NPInvalidPayloadError("feature response is not a JSON object")
    return data


class NPInvalidPayloadError(ValueError):
    """API returned malformed JSON or an unexpected envelope."""


def _density_from_document(doc: Mapping[str, Any]) -> Optional[float]:
    raw = doc.get("frac_nonzero")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _explanations_from_document(
    doc: Mapping[str, Any],
) -> tuple[ExplanationSnippet, ...]:
    items = doc.get("explanations")
    if not items or not isinstance(items, list):
        return ()
    out: list[ExplanationSnippet] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("description") or item.get("text") or "").strip()
        if not text:
            continue
        out.append(
            ExplanationSnippet(
                text=text,
                method=item.get("typeName") if isinstance(item.get("typeName"), str) else None,
                explainer_model=item.get("explanationModelName")
                if isinstance(item.get("explanationModelName"), str)
                else None,
            )
        )
    return tuple(out)


def _infer_dead_feature(density: Optional[float]) -> bool:
    if density is None:
        return False
    try:
        return float(density) <= 0.0
    except (TypeError, ValueError):
        return False


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_cache(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _load_sae_index_map(path: Path) -> dict[str, Any]:
    blob = _load_cache(path)
    if not blob or not isinstance(blob, dict):
        return {}
    if blob.get("version") != 1 or blob.get("kind") != "neuronpedia_sae_index_map":
        return {}
    items = blob.get("items")
    if not isinstance(items, dict):
        return {}
    return items


def _write_sae_index_map(
    *,
    path: Path,
    model_id: str,
    source: str,
    items: dict[str, Any],
) -> None:
    payload = {
        "version": 1,
        "kind": "neuronpedia_sae_index_map",
        "model_id": str(model_id),
        "source": str(source),
        "items": items,
    }
    _atomic_write_json(path, payload)


def _payload_to_feature_label(data: dict[str, Any], *, cached: bool) -> FeatureLabel:
    if data.get("detail") == "not_found":
        mid = str(data["model_id"])
        src = str(data["source"])
        idx = int(data["index"])
        return FeatureLabel(
            model_id=mid,
            source=src,
            index=idx,
            primary_text=NOT_FOUND_TEXT,
            explanations=(),
            density=None,
            status="not_found",
            detail="from_cache" if cached else None,
            neuronpedia_url=_feature_url(mid, src, idx),
        )

    expl = data.get("explanations") or []
    snippets = tuple(
        ExplanationSnippet(
            text=str(e.get("text", "")).strip(),
            method=e.get("method"),
            explainer_model=e.get("explainer_model"),
        )
        for e in expl
        if str(e.get("text", "")).strip()
    )
    density = data.get("density")
    if density is not None:
        density = float(density)

    primary = next((s.text for s in snippets if s.text), NO_EXPLANATION_TEXT)
    status: LabelStatus = "cached" if cached else "ok"
    if not snippets:
        status = "no_explanation" if not cached else "cached"
    if data.get("dead_feature"):
        primary = DEAD_FEATURE_TEXT if not snippets else primary
        status = "dead_feature"

    mid = str(data["model_id"])
    src = str(data["source"])
    idx = int(data["index"])
    return FeatureLabel(
        model_id=mid,
        source=src,
        index=idx,
        primary_text=primary,
        explanations=snippets,
        density=density,
        status=status,
        detail=data.get("detail"),
        neuronpedia_url=_feature_url(mid, src, idx),
    )


def _document_to_cache_payload(
    *,
    model_id: str,
    source: str,
    index: int,
    density: Optional[float],
    snippets: tuple[ExplanationSnippet, ...],
    dead_feature: bool,
    detail: Optional[str],
) -> dict[str, Any]:
    exps = [
        {"text": s.text, "method": s.method, "explainer_model": s.explainer_model} for s in snippets
    ]
    return {
        "version": 1,
        "model_id": model_id,
        "source": source,
        "index": int(index),
        "density": density,
        "explanations": exps,
        "dead_feature": dead_feature,
        "detail": detail,
    }


def fetch_feature_label(
    model_id: str,
    source: str,
    index: Union[int, str],
    *,
    api_key: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    allow_missing_key: bool = False,
    force_refresh: bool = False,
) -> FeatureLabel:
    """Fetch Neuronpedia explanations for one SAE feature with disk caching and fallbacks.

    Args:
        model_id: Neuronpedia model id (e.g. ``gemma-2-2b``).
        source: Source string (e.g. ``3-gemmascope-att-16k``).
        index: Feature index (string or int).
        api_key: Optional override; otherwise ``NEURONPEDIA_API_KEY`` / Neuronpedia global key.
        cache_dir: Defaults to ``.cache/neuronpedia_labels`` (gitignored via ``.cache``).
        use_cache: Read/write JSON cache.
        allow_missing_key: If True and no key is configured, return :attr:`FeatureLabel.status`
            ``no_api_key`` instead of raising.
        force_refresh: Skip valid cache read (still writes after a successful fetch).

    Returns:
        :class:`FeatureLabel` with ``primary_text`` safe to show even when explanations are absent.
    """
    idx_int = int(str(index).strip())
    mid = str(model_id).strip()
    src = str(source).strip()
    cache_root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cpath = cache_root / f"{_cache_key(mid, src, idx_int)}.json"
    sae_map_path = _sae_index_cache_path(cache_root, mid, src)

    url = _feature_url(mid, src, idx_int)

    if use_cache and not force_refresh:
        # Fast path: per-SAE index map.
        items = _load_sae_index_map(sae_map_path)
        hit = items.get(str(idx_int))
        if isinstance(hit, dict) and hit.get("version") == 1:
            try:
                return _payload_to_feature_label(hit, cached=True)
            except (KeyError, TypeError, ValueError):
                pass

        # Back-compat: legacy per-feature file cache.
        blob = _load_cache(cpath)
        if blob and blob.get("version") == 1:
            try:
                return _payload_to_feature_label(blob, cached=True)
            except (KeyError, TypeError, ValueError):
                pass

    env_key = os.getenv("NEURONPEDIA_API_KEY")
    effective_key = api_key if api_key not in (None, "") else env_key
    if not effective_key:
        if allow_missing_key:
            return FeatureLabel(
                model_id=mid,
                source=src,
                index=idx_int,
                primary_text=NO_EXPLANATION_TEXT,
                explanations=(),
                density=None,
                status="no_api_key",
                detail="Set NEURONPEDIA_API_KEY or pass api_key=.",
                neuronpedia_url=url,
            )
        raise NPKeyMissingError(
            "No API key provided. Set NEURONPEDIA_API_KEY or pass api_key= to fetch_feature_label."
        )

    key_used = str(api_key).strip() if api_key not in (None, "") else str(env_key or "").strip()

    try:
        doc = _fetch_feature_document(mid, src, idx_int, key_used)
    except NPRateLimitError as e:
        if use_cache:
            stale = _load_cache(cpath)
            if stale and stale.get("version") == 1:
                fl = _payload_to_feature_label(stale, cached=True)
                return FeatureLabel(
                    model_id=fl.model_id,
                    source=fl.source,
                    index=fl.index,
                    primary_text=fl.primary_text,
                    explanations=fl.explanations,
                    density=fl.density,
                    status="cached",
                    detail=f"Rate limited; using stale cache. ({e})",
                    neuronpedia_url=fl.neuronpedia_url,
                )
        return FeatureLabel(
            model_id=mid,
            source=src,
            index=idx_int,
            primary_text=GENERIC_ERROR_TEXT,
            explanations=(),
            density=None,
            status="rate_limited",
            detail=str(e),
            neuronpedia_url=url,
        )
    except NPUnauthorizedError as e:
        return FeatureLabel(
            model_id=mid,
            source=src,
            index=idx_int,
            primary_text=GENERIC_ERROR_TEXT,
            explanations=(),
            density=None,
            status="unauthorized",
            detail=str(e),
            neuronpedia_url=url,
        )
    except requests.exceptions.HTTPError as e:
        if getattr(e, "response", None) is not None and e.response is not None:
            if e.response.status_code == 404:
                fl = FeatureLabel(
                    model_id=mid,
                    source=src,
                    index=idx_int,
                    primary_text=NOT_FOUND_TEXT,
                    explanations=(),
                    density=None,
                    status="not_found",
                    detail="HTTP 404 from Neuronpedia API.",
                    neuronpedia_url=url,
                )
                if use_cache:
                    payload = {
                        "version": 1,
                        "model_id": mid,
                        "source": src,
                        "index": idx_int,
                        "density": None,
                        "explanations": [],
                        "dead_feature": False,
                        "detail": "not_found",
                    }
                    _atomic_write_json(cpath, payload)
                return fl
        return FeatureLabel(
            model_id=mid,
            source=src,
            index=idx_int,
            primary_text=GENERIC_ERROR_TEXT,
            explanations=(),
            density=None,
            status="error",
            detail=str(e),
            neuronpedia_url=url,
        )
    except NPInvalidPayloadError as e:
        return FeatureLabel(
            model_id=mid,
            source=src,
            index=idx_int,
            primary_text=GENERIC_ERROR_TEXT,
            explanations=(),
            density=None,
            status="error",
            detail=str(e),
            neuronpedia_url=url,
        )
    except Exception as e:  # noqa: BLE001
        return FeatureLabel(
            model_id=mid,
            source=src,
            index=idx_int,
            primary_text=GENERIC_ERROR_TEXT,
            explanations=(),
            density=None,
            status="error",
            detail=repr(e),
            neuronpedia_url=url,
        )

    density = _density_from_document(doc)
    snippets = _explanations_from_document(doc)
    dead = _infer_dead_feature(density)
    if not snippets:
        primary = DEAD_FEATURE_TEXT if dead else NO_EXPLANATION_TEXT
        status_ok: LabelStatus = "dead_feature" if dead else "no_explanation"
    else:
        primary = snippets[0].text
        status_ok = "dead_feature" if dead else "ok"

    payload = _document_to_cache_payload(
        model_id=mid,
        source=src,
        index=idx_int,
        density=density,
        snippets=snippets,
        dead_feature=dead,
        detail=None,
    )
    if use_cache:
        # Write both: per-feature legacy file + per-SAE map for efficient batch lookup.
        _atomic_write_json(cpath, payload)
        try:
            items = _load_sae_index_map(sae_map_path)
            items[str(idx_int)] = payload
            _write_sae_index_map(path=sae_map_path, model_id=mid, source=src, items=items)
        except Exception:
            # Best-effort; single-feature cache still works.
            pass

    return FeatureLabel(
        model_id=mid,
        source=src,
        index=idx_int,
        primary_text=primary,
        explanations=snippets,
        density=density,
        status=status_ok,
        detail=None,
        neuronpedia_url=url,
    )


def fetch_feature_labels_batch(
    keys: Sequence[tuple[str, str, Union[int, str]]],
    **kwargs: Any,
) -> list[FeatureLabel]:
    """Fetch many ``(model_id, source, index)`` tuples.

    This function is optimized for the common research workflow:
    - You have a large list of latents (e.g. 500 indices) from one SAE (same ``model_id`` + ``source``).
    - You want labels quickly without 500 separate API round-trips on repeat runs.

    Behavior:
    - Uses the per-SAE `index_map.v1.json` cache for fast local lookups (keyed by ``(model_id, source, index)``).
    - Falls back to the legacy per-feature cache files if present.
    - Only hits the Neuronpedia API for missing indices, then backfills both caches.
    """

    # Preserve ordering and allow mixed SAEs; group by (model_id, source) for map-cache efficiency.
    out: list[Optional[FeatureLabel]] = [None] * len(keys)

    cache_dir = kwargs.get("cache_dir")
    cache_root = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    use_cache = bool(kwargs.get("use_cache", True))
    force_refresh = bool(kwargs.get("force_refresh", False))

    # First pass: try to satisfy from per-SAE map (and per-feature legacy).
    misses: list[tuple[int, str, str, int]] = []
    sae_map_cache: dict[tuple[str, str], dict[str, Any]] = {}
    if use_cache and not force_refresh:
        for j, (m, s, i) in enumerate(keys):
            mid = str(m).strip()
            src = str(s).strip()
            idx_int = int(str(i).strip())
            map_key = (mid, src)
            if map_key not in sae_map_cache:
                sae_map_cache[map_key] = _load_sae_index_map(_sae_index_cache_path(cache_root, mid, src))
            hit = sae_map_cache[map_key].get(str(idx_int))
            if isinstance(hit, dict) and hit.get("version") == 1:
                try:
                    out[j] = _payload_to_feature_label(hit, cached=True)
                    continue
                except Exception:
                    pass
            # legacy fallback
            legacy = _load_cache(cache_root / f"{_cache_key(mid, src, idx_int)}.json")
            if isinstance(legacy, dict) and legacy.get("version") == 1:
                try:
                    out[j] = _payload_to_feature_label(legacy, cached=True)
                    continue
                except Exception:
                    pass
            misses.append((j, mid, src, idx_int))

    else:
        for j, (m, s, i) in enumerate(keys):
            misses.append((j, str(m).strip(), str(s).strip(), int(str(i).strip())))

    # Second pass: fetch misses (still uses cache writes inside fetch_feature_label).
    for j, mid, src, idx_int in misses:
        out[j] = fetch_feature_label(mid, src, idx_int, **kwargs)

    # Safety: should be fully populated now.
    return [x for x in out if x is not None]
