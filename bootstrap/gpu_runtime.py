"""Optional CUDA environment; the launcher and CPU worker remain CPU-only."""
from pathlib import Path
import json
import os
import shutil
import subprocess

from portable import atomic_json, digest, prepare_runtime, read_json, safe_path, status

PROBE = """import json, torch
from ffxiv_prism.runtime_contract import validate_runtime
validate_runtime()
assert torch.cuda.is_available(), 'CUDA 不可用，请检查显卡驱动'
x = torch.ones((16, 16), device='cuda')
(x @ x).sum().item()
torch.nn.functional.conv2d(torch.ones((1, 3, 16, 16), device='cuda'), torch.ones((4, 3, 3, 3), device='cuda')).sum().item()
torch.cuda.synchronize()
print(json.dumps({'available': True, 'label': 'GPU', 'device_name': torch.cuda.get_device_name(0), 'reason': None}))
"""


def settings(home):
    path = Path(home) / 'user/runtime.json'
    return read_json(path) if path.exists() else {}


def choose_variant(rows):
    """Select the first visible GPU, matching torch's default cuda:0."""
    row = rows.strip().splitlines()[0].split(',')
    capability = tuple(map(int, row[0].strip().split('.')))
    driver = tuple(map(int, row[1].strip().split('.')))
    if capability >= (7, 5) and driver >= (580, 88):
        return 'cu130'
    if (5, 0) <= capability < (10, 0) and driver >= (560, 76):
        return 'cu126'
    raise ValueError('当前显卡架构或驱动不满足 GPU 环境要求；请更新 NVIDIA 驱动或使用 CPU。')


def detect_variant():
    result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap,driver_version',
                             '--format=csv,noheader,nounits'], capture_output=True,
                            text=True, check=True, timeout=20)
    return choose_variant(result.stdout)


def worker_python(home, app, variant):
    if variant not in ('cu126', 'cu130'):
        raise ValueError('未知 GPU 环境')
    lock_path = Path(app) / f'manifests/runtime-{variant}.json'
    lock = read_json(lock_path)
    env = safe_path(home, 'runtime/' + lock['id'])
    marker = env / 'ready.json'
    if not marker.exists() or read_json(marker).get('lock_sha256') != digest(lock_path):
        raise ValueError('GPU 依赖尚未准备完成，请重新运行安装环境.bat')
    interpreter = env / 'python'
    interpreter.mkdir(parents=True, exist_ok=True)
    for source in (Path(home) / 'runtime/python').iterdir():
        if source.is_file() and source.suffix != '._pth':
            target = interpreter / source.name
            if not target.exists():
                shutil.copy2(source, target)
    paths = ['python312.zip', '.', '../site-packages',
             os.path.relpath(Path(app) / 'src', interpreter), os.path.relpath(app, interpreter),
             os.path.relpath(Path(home) / 'bootstrap', interpreter), 'import site']
    (interpreter / 'python312._pth').write_text('\n'.join(paths) + '\n', encoding='utf-8')
    return interpreter / 'python.exe'


def probe(python):
    result = subprocess.run([str(python), '-X', 'utf8', '-c', PROBE],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-2000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def prepare_optional(home, app, requested=None):
    config = settings(home)
    if requested == 'cpu':
        # Choosing CPU never deletes a previously installed GPU environment.
        atomic_json(Path(home) / 'user/runtime.json', config)
        return
    if requested != 'gpu' and not config.get('gpu'):
        return
    try:
        variant = detect_variant() if requested == 'gpu' else config['gpu']
        if variant not in ('cu126', 'cu130'):
            raise ValueError('未知 GPU 环境，请重新选择安装环境。')
        status(f'额外安装 GPU 环境：{variant}；CPU 环境保留。')
        prepare_runtime(home, app, f'runtime-{variant}.json')
        python = worker_python(home, app, variant)
        probe(python)
        atomic_json(Path(home) / 'user/runtime.json', {'gpu': variant})
        status('GPU 环境自检通过。')
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        status(f'GPU 环境未就绪：{exc}\nCPU 环境仍可使用；重跑安装环境.bat 可继续追加 GPU。')
        return False
    return True


def expose_optional(home, app):
    os.environ.pop('PRISM_GPU_PYTHON', None)
    variant = settings(home).get('gpu')
    if variant:
        try:
            os.environ['PRISM_GPU_PYTHON'] = str(worker_python(home, app, variant))
        except (OSError, ValueError, KeyError) as exc:
            status(f'GPU 环境未就绪，继续使用 CPU：{exc}')


def installation_choice():
    print('1. 安装cpu环境（推荐）\n2. 额外安装gpu环境（cpu环境+gpu环境）', flush=True)
    while True:
        try:
            value = input('请选择 [1/2，默认 1]：').strip()
        except EOFError:
            value = '1'
        if value in ('', '1', '2'):
            return 'gpu' if value == '2' else 'cpu'
        print('请输入 1 或 2。', flush=True)
