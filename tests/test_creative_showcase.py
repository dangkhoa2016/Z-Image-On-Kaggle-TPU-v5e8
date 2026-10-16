"""Reproducible source-level checks for the eight-image creative showcase."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "showcase"


def test_creative_prompt_suite_is_unique_and_deterministic():
    rows = json.loads((SCRIPTS / "creative-prompts.json").read_text())
    assert len(rows) == 8
    assert [row["seed"] for row in rows] == [501, 502, 503, 504, 615, 506, 507, 508]
    assert len({row["id"] for row in rows}) == 8
    assert len({row["prompt"] for row in rows}) == 8
    assert all(row["negative_prompt"] for row in rows)
    assert rows[4]["seed"] == 615
    assert "strap" in rows[4]["prompt"].lower()
    assert "strap" in rows[4]["negative_prompt"].lower()
    assert (ROOT/"assets/showcase-photoreal-watch-final.png").is_file()


def test_creative_pipeline_is_present_and_syntax_valid():
    needed = ["prepare_prompts.py", "encode_all_optimized.py",
              "run_creative.py", "decode_creative_batch.py", "run_zimage_creative.py",
              "gallery_runtime.py", "serve_gallery.py"]
    for name in needed:
        text = (SCRIPTS / name).read_text()
        compile(text, name, "exec")
        assert not re.search(r"\\b(?:K4|R6|P8)\\b", text, re.IGNORECASE)


def test_notebook_runs_real_eight_image_pipeline():
    notebook = json.loads((ROOT / "notebooks/kaggle-tpu-v5e8-production.ipynb").read_text())
    assert len(notebook["cells"]) == 48
    code = "\\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "run_creative.py" in code
    assert "encode_all_optimized.py" in code
    assert 'expected_count = 15 if RUN_2000PX else 14' in code
    assert "serve_gallery.py" in code
    assert "git" in code and "clone" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "production-notebook", "exec")
            assert cell["execution_count"] is None
            assert not cell.get("outputs")


def test_eight_prompts_two_four_replica_batches_one_vae_load():
    runner = (SCRIPTS / "run_creative.py").read_text()
    encoder = (SCRIPTS / "encode_all_optimized.py").read_text()
    decoder = (SCRIPTS / "decode_creative_batch.py").read_text()
    assert '("creative-first",[0,1,2,3])' in runner
    assert '("creative-second",[4,5,6,7])' in runner
    assert '"--replicas","4"' in runner
    assert '"--batch-size","4"' in runner
    assert 'len(items)==8' in encoder
    assert "for i,x in enumerate(items):" in encoder
    assert decoder.count('AutoencoderKL.from_pretrained(') == 1
    assert 'fresh-six-manifest.json' in runner
    assert 'creative-extra-acceptance.json' in runner
    assert 'len(set(x["sha256"] for x in rows))==8' in runner


def test_gallery_defaults_and_fresh_evidence_contract():
    runtime = (SCRIPTS / "gallery_runtime.py").read_text()
    notebook = json.loads((ROOT / "notebooks/kaggle-tpu-v5e8-production.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "expected_count=15 if high_enabled else 14" in runtime
    assert "run_highres" in runtime
    assert "publication-512-alternate-seed43/output.png" in runtime
    assert "provenance='FRESH_INFERENCE'" in runtime
    assert "showcase-creative-first.json" in runtime
    assert "showcase-creative-second.json" in runtime
    assert "current-run" not in code or "RUN_ID" in code
    assert "SECOND_INDEPENDENT_512_PASS" in code
    assert "expected_count = 15 if RUN_2000PX else 14" in code

def test_published_docs_match_current_batches():
    en = (ROOT / "docs/CREATIVE_SHOWCASE.md").read_text()
    vi = (ROOT / "docs/CREATIVE_SHOWCASE.vi.md").read_text()
    assert "two real batches" in en
    assert "hai batch TPU" in vi
    assert "ZIMAGE_RUN_2000PX=True" in en
    assert "ZIMAGE_RUN_2000PX=True" in vi
    for discarded in ("run_six.py", "run_extra.py", "encode_extra.py", "build_gallery.py"):
        assert not (SCRIPTS / discarded).exists()


def test_gallery_empty_state_has_actionable_bilingual_controls():
    template = (SCRIPTS / "gallery_template.html").read_text()
    assert 'id="none" class="emptyState hidden"' in template
    assert 'id="emptyAll"' in template
    assert 'id="empty1024"' in template
    assert 'aria-live="polite"' in template
    assert 'emptyHighres:' in template
    assert 'Không có ảnh phù hợp với bộ lọc' in template
    assert 'updateEmptyState()' in template
