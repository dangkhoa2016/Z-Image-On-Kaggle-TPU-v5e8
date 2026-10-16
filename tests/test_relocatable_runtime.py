from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_single_inference_cache_follows_checkout():
    from run_zimage_spmd import build_arg_parser
    args = build_arg_parser().parse_args([
        '--width', '512', '--height', '512', '--slug', 'portable',
        '--embeddings', '/tmp/prompt.pt',
    ])
    assert args.xla_cache_dir == str(ROOT / 'xla-cache')


def test_batch_config_follows_checkout():
    from scripts.run_zimage_tpu_batch4 import describe_config
    config = describe_config()
    assert config['xla_cache_dir'] == str(ROOT / 'xla-cache-tpu-batch4')


def test_scripts_do_not_write_into_other_checkout():
    targets = [
        'run_zimage_spmd.py', 'scripts/decode_vae.py',
        'scripts/run_zimage_tpu_batch4.py',
        'scripts/showcase/run_creative.py',
        'scripts/showcase/run_zimage_creative.py',
        'scripts/showcase/decode_creative_batch.py',
        'scripts/showcase/encode_all_optimized.py',
    ]
    for name in targets:
        assert '/kaggle/working/zimage-tpu' not in (ROOT / name).read_text()


def test_batch_describe_config_from_arbitrary_working_directory(tmp_path):
    runner = ROOT / 'scripts/run_zimage_tpu_batch4.py'
    proc = subprocess.run([sys.executable, str(runner), '--describe-config'],
                          cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert str(ROOT / 'xla-cache-tpu-batch4') in proc.stdout
