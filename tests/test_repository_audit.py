from pathlib import Path

import pytest

from scripts.repository_audit import audit_repository


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "README.md").write_text(
        "# Demo\n\n"
        "[![CI](https://github.com/example/demo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/example/demo/actions/workflows/ci.yml)\n\n"
        "> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)\n"
    )
    (tmp_path / "README.vi.md").write_text(
        "# Demo VI\n\n"
        "> 🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**\n"
    )
    (tmp_path / "docs" / "architecture.md").write_text(
        "# Architecture\n\n"
        "> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](architecture.vi.md)\n"
    )
    (tmp_path / "docs" / "architecture.vi.md").write_text(
        "# Kiến trúc\n\n"
        "> 🌐 Ngôn ngữ / Language: [English](architecture.md) | **Tiếng Việt**\n"
    )
    (tmp_path / "LICENSE-NOTES.md").write_text("Repository code is MIT. Upstream model rights remain upstream.\n")
    (tmp_path / "LICENSE-NOTES.vi.md").write_text("Mã nguồn dự án dùng MIT. Quyền upstream vẫn thuộc upstream.\n")
    return tmp_path


def test_reports_missing_vietnamese_pair(tmp_path: Path):
    root = make_repo(tmp_path)
    (root / "docs" / "architecture.vi.md").unlink()
    findings = audit_repository(root)
    assert any("missing bilingual pair" in item for item in findings)


def test_reports_broken_reciprocal_language_link(tmp_path: Path):
    root = make_repo(tmp_path)
    (root / "docs" / "architecture.vi.md").write_text("# Kiến trúc\n")
    findings = audit_repository(root)
    assert any("missing reciprocal language switch" in item for item in findings)


def test_reports_static_ci_badge(tmp_path: Path):
    root = make_repo(tmp_path)
    (root / "README.md").write_text(
        "# Demo\n\n"
        "[![CI](https://img.shields.io/badge/CI-passing-success)](#validation)\n\n"
        "> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)\n"
    )
    findings = audit_repository(root)
    assert any("static CI badge" in item for item in findings)


def test_reports_prohibited_model_relicensing_wording(tmp_path: Path):
    root = make_repo(tmp_path)
    (root / "LICENSE-NOTES.md").write_text("The Z-Image model weights are relicensed under MIT by this repository.\n")
    findings = audit_repository(root)
    assert any("model relicensing claim" in item for item in findings)


def test_reports_missing_github_markdown_pair(tmp_path: Path):
    root = make_repo(tmp_path)
    (root / ".github" / "SECURITY.md").write_text("# Security\n")
    findings = audit_repository(root)
    assert any(".github/SECURITY.md" in item and "missing bilingual pair" in item for item in findings)


def test_reports_broken_readme_language_switch(tmp_path: Path):
    root = make_repo(tmp_path)
    (root / "README.vi.md").write_text("# Demo VI\n")
    findings = audit_repository(root)
    assert any("README.md <-> README.vi.md" in item and "missing reciprocal language switch" in item for item in findings)



def test_reports_transient_actions_run_url_in_public_release_note(tmp_path: Path):
    root = make_repo(tmp_path)
    releases = root / "docs" / "releases"
    releases.mkdir(parents=True)
    (releases / "v1.0.0.md").write_text(
        "# Release\n\n> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](v1.0.0.vi.md)\n\n"
        "https://github.com/example/demo/actions/runs/12345\n"
    )
    (releases / "v1.0.0.vi.md").write_text(
        "# Release VI\n\n> 🌐 Ngôn ngữ / Language: [English](v1.0.0.md) | **Tiếng Việt**\n"
    )
    findings = audit_repository(root)
    assert any("transient Actions run URL" in item for item in findings)


def test_allows_clean_canonical_notebook_template(tmp_path: Path):
    root = make_repo(tmp_path)
    notebooks = root / "notebooks"
    notebooks.mkdir()
    (notebooks / "kaggle-tpu-v5e8-production.ipynb").write_text(
        '{"cells": [{"cell_type": "code", "execution_count": null, "outputs": [], "source": []}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
    )
    findings = audit_repository(root)
    assert not any("notebook" in item.lower() for item in findings)


def test_rejects_unexpected_versioned_notebook(tmp_path: Path):
    root = make_repo(tmp_path)
    notebooks = root / "notebooks"
    notebooks.mkdir()
    (notebooks / "demo.ipynb").write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}')
    findings = audit_repository(root)
    assert any("unexpected versioned notebook" in item for item in findings)


def test_rejects_execution_state_in_canonical_notebook_template(tmp_path: Path):
    root = make_repo(tmp_path)
    notebooks = root / "notebooks"
    notebooks.mkdir()
    (notebooks / "kaggle-tpu-v5e8-production.ipynb").write_text(
        '{"cells": [{"cell_type": "code", "execution_count": 1, "outputs": [{"output_type": "stream", "name": "stdout", "text": ["x"]}], "source": []}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
    )
    findings = audit_repository(root)
    assert any("contains execution state" in item for item in findings)


def test_rejects_single_role_audience_framing(tmp_path: Path):
    root = make_repo(tmp_path)
    (root / "README.md").write_text(
        "# Demo\n\n"
        "> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)\n\n"
        "This " + "review" + "er-facing repository is intended for " + "review" + "ers.\n"
    )
    findings = audit_repository(root)
    assert any("single-role audience framing" in item for item in findings)


def test_rejects_internal_stage_tokens_in_published_filenames(tmp_path: Path):
    root = make_repo(tmp_path)
    stage_name = "D" + str(10) + "-REPORT.md"
    (root / stage_name).write_text("# Internal experiment\n")
    findings = audit_repository(root)
    assert any("internal stage token in public filename" in x for x in findings)
