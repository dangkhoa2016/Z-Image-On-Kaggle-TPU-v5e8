from pathlib import Path
import json
import re


def _paired_markdown_files(root: Path):
    for path in root.rglob("*.md"):
        rel = path.relative_to(root)
        if any(part.startswith(".") and part != ".github" for part in rel.parts):
            continue
        if path.name in {"LICENSE", "README.md", "README.vi.md"}:
            continue
        if path.name.endswith(".vi.md"):
            yield path, path.with_name(path.name[:-6] + ".md"), "vi"
        else:
            yield path, path.with_name(path.stem + ".vi.md"), "en"


def audit_repository(root: Path) -> list[str]:
    root = Path(root)
    findings: list[str] = []

    # Publicly shipped filenames must describe their purpose instead of
    # using internal experiment-stage identifiers (e.g. D10, K3, S4).
    stage_filename = re.compile(r"(?<![A-Za-z0-9])[DKRPSM][0-9]{1,3}(?![A-Za-z0-9])", re.I)
    for candidate in root.rglob("*"):
        if not candidate.is_file() or ".git" in candidate.parts:
            continue
        relative = candidate.relative_to(root)
        if any(part.startswith(".") and part != ".github" for part in relative.parts):
            continue
        if stage_filename.search(candidate.name):
            findings.append(f"internal stage token in public filename: {relative}")

    seen = set()
    for path, pair, lang in _paired_markdown_files(root):
        key = tuple(sorted((str(path), str(pair))))
        if key in seen:
            continue
        seen.add(key)
        if not pair.exists():
            findings.append(f"missing bilingual pair: {path.relative_to(root)}")
            continue
        en = path if lang == "en" else pair
        vi = pair if lang == "en" else path
        en_text = en.read_text(errors="replace")
        vi_text = vi.read_text(errors="replace")
        if vi.name not in en_text or en.name not in vi_text:
            findings.append(f"missing reciprocal language switch: {en.relative_to(root)} <-> {vi.relative_to(root)}")

    readme = root / "README.md"
    readme_vi = root / "README.vi.md"
    if readme.exists() != readme_vi.exists():
        findings.append("missing bilingual pair: README.md <-> README.vi.md")
    elif readme.exists() and readme_vi.exists():
        text = readme.read_text(errors="replace")
        vi_text = readme_vi.read_text(errors="replace")
        if "README.vi.md" not in text or "README.md" not in vi_text:
            findings.append("missing reciprocal language switch: README.md <-> README.vi.md")
        if "img.shields.io/badge/CI-" in text or "img.shields.io/badge/CI%20" in text:
            findings.append("static CI badge detected in README.md")

    for candidate in [root / "LICENSE-NOTES.md", root / "docs" / "MODEL_LICENSE.md"]:
        if candidate.exists():
            text = candidate.read_text(errors="replace")
            if re.search(r"model weights?.{0,40}relicensed under MIT|relicensed under MIT.{0,40}model weights?", text, re.I | re.S):
                findings.append(f"model relicensing claim detected: {candidate.relative_to(root)}")

    # Git ships exactly one clean Kaggle notebook template. It must remain
    # unexecuted: no execution_count and no cell outputs are allowed.
    allowed_notebook = Path("notebooks/kaggle-tpu-v5e8-production.ipynb")
    notebooks = [candidate for candidate in root.rglob("*.ipynb") if ".git" not in candidate.parts]
    for candidate in notebooks:
        rel = candidate.relative_to(root)
        if rel != allowed_notebook:
            findings.append(f"unexpected versioned notebook: {rel}")
            continue
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            findings.append(f"invalid notebook template: {rel}")
            continue
        for index, cell in enumerate(payload.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs", []):
                findings.append(f"notebook template contains execution state: {rel} cell {index}")
                break

    # Public project materials must describe the repository as a community-facing
    # research/engineering project rather than centering it on one evaluation role.
    audience_pattern = re.compile(r"\breview" + r"ers?\b", re.I)
    for candidate in root.rglob("*.md"):
        rel = candidate.relative_to(root)
        if any(part.startswith(".") and part != ".github" for part in rel.parts):
            continue
        if audience_pattern.search(candidate.read_text(errors="replace")):
            findings.append(f"single-role audience framing detected: {rel}")



    releases = root / "docs" / "releases"
    if releases.exists():
        for candidate in releases.glob("*.md"):
            if "/actions/runs/" in candidate.read_text(errors="replace"):
                findings.append(f"transient Actions run URL detected: {candidate.relative_to(root)}")

    # Publication boundary: public prose and tracked reference JSON must never
    # reveal private evaluation sequences or reviewer decision records.
    prohibited = re.compile(
        r"PASS_WITH_REMEDIATION|first[-_ ]pass|remediation[-_ ]v?2|"
        r"rejected[_ ]originals|approval[-_ ]decision|superseded[_ ]seed|"
        r"strapless render|thiếu dây|human[-_ ]approved replacement|"
        r"reconstructed development chronology", re.I
    )
    for candidate in [*root.glob("*.md"), *root.glob("docs/**/*.md"),
                      *root.glob("evidence/reference/*.json")]:
        if candidate.exists() and prohibited.search(candidate.read_text(errors="replace")):
            findings.append(f"internal publication detail detected: {candidate.relative_to(root)}")
    for forbidden in (root / "evidence/dual", root / "evidence/publication",
                      root / "evidence/visual-remediation"):
        if forbidden.exists() and any(forbidden.iterdir()):
            # Runtime output directories are intentionally created during TPU execution,
            # so only check files tracked by Git, not transient runtime outputs.
            import subprocess
            tracked = subprocess.run(["git", "ls-files", "--", str(forbidden.relative_to(root))],
                         cwd=root, capture_output=True, text=True)
            if tracked.returncode == 0 and tracked.stdout.strip():
                findings.append(f"internal evidence tracked: {forbidden.relative_to(root)}")

    # Documentation integrity checks run on every CI build, not just at release time.
    import urllib.parse
    tracked = set()
    for item in root.rglob("*"):
        if item.is_file() and ".git" not in item.parts:
            tracked.add(item.relative_to(root).as_posix())
    md_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for document in root.rglob("*.md"):
        rel_doc = document.relative_to(root)
        if any(part.startswith(".") and part != ".github" for part in rel_doc.parts):
            continue
        body = document.read_text(errors="replace")
        for match in md_pattern.finditer(body):
            url = match.group(1).strip().split()[0].strip("<>")
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme in ("http", "https"):
                if parsed.netloc.lower() == "github.com":
                    prefix = "/dangkhoa2016/Z-Image-On-Kaggle-TPU-v5e8/actions/workflows/"
                    if parsed.path.startswith(prefix):
                        name = parsed.path[len(prefix):].split("/")[0]
                        if ".github/workflows/" + name not in tracked:
                            findings.append(f"missing GitHub workflow: {rel_doc}: {url}")
                continue
            if parsed.scheme or url.startswith(("#", "//", "mailto:")):
                continue
            target = (document.parent / urllib.parse.unquote(parsed.path)).resolve()
            try:
                target.relative_to(root.resolve())
            except ValueError:
                findings.append(f"documentation link outside repository: {rel_doc}: {url}")
                continue
            if not target.exists():
                findings.append(f"broken documentation link: {rel_doc}: {url}")
        for old_path in ("evidence/publication/", "evidence/dual/"):
            if old_path in body and rel_doc.as_posix().startswith("docs/PROJECT_GUIDE"):
                findings.append(f"obsolete evidence directory in guide: {rel_doc}: {old_path}")
        if "[Final Acceptance Report](" in body and "evidence/reference/production.json" in body:
            findings.append(f"misleading reference-metric link label: {rel_doc}")

    return findings


def main() -> int:
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    findings = audit_repository(root)
    if findings:
        for item in findings:
            print(f"FAIL: {item}")
        return 1
    print("repository audit: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
