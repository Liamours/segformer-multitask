from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PairedImageSample:
    case_id: str
    anterior_path: Path
    posterior_path: Path
    views: tuple[str, str] = ("anterior", "posterior")


VIEW_SUFFIX_PATTERN = re.compile(r"(_0000|_0001|_anterior|_posterior|_ant|_post)$", re.IGNORECASE)


def infer_case_id(path: str | Path) -> str:
    stem = Path(path).stem
    case_id = VIEW_SUFFIX_PATTERN.sub("", stem)
    if not case_id:
        raise ValueError(f"Could not infer case id from path={path}.")
    return case_id


def build_pair(anterior_path: str | Path, posterior_path: str | Path) -> PairedImageSample:
    anterior = Path(anterior_path)
    posterior = Path(posterior_path)
    if not anterior.is_file():
        raise FileNotFoundError(f"Anterior image not found: {anterior}")
    if not posterior.is_file():
        raise FileNotFoundError(f"Posterior image not found: {posterior}")

    anterior_case = infer_case_id(anterior)
    posterior_case = infer_case_id(posterior)
    if anterior_case != posterior_case:
        raise ValueError(f"Anterior/posterior case mismatch: {anterior_case} != {posterior_case}.")
    return PairedImageSample(case_id=anterior_case, anterior_path=anterior, posterior_path=posterior)
