from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "docs/releases/v1.0.0.md"
VI = ROOT / "docs/releases/v1.0.0.vi.md"


def test_release_notes_exist_as_bilingual_pair_with_language_switches():
    assert EN.exists()
    assert VI.exists()
    en = EN.read_text()
    vi = VI.read_text()
    assert "v1.0.0.vi.md" in en
    assert "v1.0.0.md" in vi


def test_release_notes_cover_publication_grade_sections():
    en = EN.read_text()
    for heading in [
        "## First public release",
        "## What v1.0.0 establishes",
        "## Production configuration",
        "## Resolution and memory qualification",
        "## Historical topology evidence",
        "## Visual acceptance",
        "## Reproducibility contract",
        "## Release assets",
        "## Quick start",
        "## Scope and licensing",
    ]:
        assert heading in en


def test_release_notes_use_stable_workflow_links_not_run_urls():
    for path in [EN, VI]:
        text = path.read_text()
        assert "/actions/runs/" not in text
        assert "/actions/workflows/ci.yml" in text
        assert "/actions/workflows/repository-audit.yml" in text


def test_release_note_pins_production_topology_and_preserves_historical_pair_evidence():
    en = EN.read_text()
    assert "(replica=4, model=2)" in en
    assert "global batch 4" in en
    assert "maximum safe tested batch" in en.lower()
    assert "90.264" in en and "0.04431" in en
    assert "(replica=2, model=4)" in en
    assert "126.264" in en and "95.001" in en and "1.329×" in en
    assert "288.645" in en and "127.218" in en and "2.269×" in en


def test_release_note_claims_fresh_canonical_tpu_execution():
    en = EN.read_text().lower()
    assert "fresh kaggle tpu execution" in en
    assert "four concurrent 1024" in en
    assert "90.099" in en
    assert "38-cell" not in en
    assert "19 code cells" not in en



def test_readmes_describe_verified_runtime_without_unpublished_demo_claims():
    en = (ROOT / "README.md").read_text()
    vi = (ROOT / "README.vi.md").read_text()
    for text in [en, vi]:
        assert "(replica=4, model=2)" in text
        assert "global batch 4" in text.lower()
        assert "maximum safe tested batch" in text.lower()
        assert "notebooks/kaggle-tpu-v5e8-production.ipynb" in text
        assert "kaggle-tpu-v5e8-production-executed.ipynb" not in text
        assert "public kaggle production demo" not in text.lower()
        assert "production demo notebook" not in text.lower()



def test_repository_ships_one_clean_production_notebook_template():
    import json
    notebook_path = ROOT / "notebooks/kaggle-tpu-v5e8-production.ipynb"
    assert notebook_path.exists()
    assert list(ROOT.rglob("*.ipynb")) == [notebook_path]
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
    reference = json.loads((ROOT / "evidence/reference/showcase.json").read_text())
    assert len(reference["images"]) == 6


def test_release_contract_uses_three_canonical_assets():
    en = EN.read_text()
    assert "Z-Image-On-Kaggle-TPU-v5e8-v1.0.0-runtime-evidence.tar.gz" in en
    assert "Z-Image-On-Kaggle-TPU-v5e8-v1.0.0-throughput-comparison.json" in en
    assert "Z-Image-On-Kaggle-TPU-v5e8-v1.0.0-showcase-stability-acceptance.json" in en
    assert "RUNTIME-SHA256SUMS.txt" not in en
    assert "CURRENT-SHA256SUMS.txt" not in en


def test_public_docs_do_not_imply_hidden_notebook_artifacts():
    paths = [ROOT / "README.md", ROOT / "README.vi.md", ROOT / "docs/KAGGLE.md", ROOT / "docs/KAGGLE.vi.md", ROOT / "docs/REPRODUCIBILITY.md", ROOT / "docs/REPRODUCIBILITY.vi.md", EN, VI]
    text = "\n".join(path.read_text().lower() for path in paths)
    for phrase in [
        "executed notebook",
        "notebook outputs",
        "notebook artifact",
        "public kaggle production demo",
        "deliverable công khai trên kaggle",
        "intentionally versions no `.ipynb`",
        "git chủ động không version `.ipynb`",
        "contains no `.ipynb` file",
        "không chứa file `.ipynb`",
    ]:
        assert phrase not in text


def test_release_notes_do_not_claim_a_published_notebook_or_demo():
    text = (EN.read_text() + "\n" + VI.read_text()).lower()
    for phrase in [
        "production demo notebook",
        "public kaggle production demo",
        "deliverable công khai trên kaggle",
        "production demo công khai trên kaggle",
        "runnable notebook",
    ]:
        assert phrase not in text
