#!/usr/bin/env python3
"""Verify resources already installed in the official image; never downloads them."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_entries(value, label):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if 'path' in value:
            return [value]
        rows = []
        for path, detail in value.items():
            if not isinstance(detail, dict):
                raise ValueError(label + ' mapping entries must be objects')
            row = dict(detail)
            row.setdefault('path', path)
            rows.append(row)
        return rows
    raise ValueError(label + ' must be a file object, list, or path mapping')


def controlled_path(value):
    if not isinstance(value, str): raise ValueError('manifest path must be text')
    # The exported large_model row omits the first slash. Accept only this
    # explicit root-relative spelling; never resolve arbitrary relative paths.
    if value.startswith('root/'):
        value = '/' + value
    path = PurePosixPath(value)
    allowed_prefixes = ('/root/EAICON/', '/root/inference/FM9G4B-V/', '/root/ultralytics/')
    allowed_files = {'/root/jaka/best.pt', '/root/isaacsim/VERSION'}
    if not path.is_absolute() or '..' in path.parts or not (value.startswith(allowed_prefixes) or value in allowed_files):
        raise ValueError('manifest path is outside the explicit image-resource scope: ' + value)
    return str(path)


def verify(entry, hash_content):
    if not isinstance(entry, dict): raise ValueError('file entry must be an object')
    path = Path(controlled_path(entry['path']))
    size = entry.get('size', entry.get('size_bytes', entry.get('bytes')))
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError('manifest file size must be a nonnegative integer: ' + str(path))
    expected_hash = entry.get('sha256', '')
    if not isinstance(expected_hash, str) or not re.fullmatch('[0-9a-fA-F]{64}', expected_hash):
        raise ValueError('manifest SHA256 must contain 64 hex characters: ' + str(path))
    result = {'path': str(path), 'expected_size': size, 'expected_sha256': expected_hash.lower(),
              'sha256_checked': hash_content, 'ok': False}
    try:
        if not path.is_file():
            result['error'] = 'missing file or broken symlink'
            return result
        result['actual_size'] = path.stat().st_size
        if result['actual_size'] != size:
            result['error'] = 'file size differs from the recorded image'
            return result
        if hash_content:
            result['actual_sha256'] = sha256(path)
            if result['actual_sha256'] != expected_hash.lower():
                result['error'] = 'SHA256 differs from the recorded image'
                return result
        result['ok'] = True
    except OSError as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parent.parent/'config/image-baseline-manifest.json')
    parser.add_argument('--deep-model', action='store_true', help='also read and hash the entire large model weight file(s)')
    parser.add_argument('--output', type=Path, help='save report to a new JSON file; refuses overwrite')
    args = parser.parse_args()
    start = time.monotonic()
    result = {'checked_at': time.time(), 'manifest': str(args.manifest.resolve()),
              'deep_model': args.deep_model, 'small_files': [], 'large_model': []}
    try:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        small = file_entries(manifest['files'], 'files')
        large = file_entries(manifest['large_model'], 'large_model')
        if not small or not large: raise ValueError('manifest must contain both files and large_model entries')
        all_paths = [controlled_path(e['path']) for e in small+large]
        if len(all_paths) != len(set(all_paths)): raise ValueError('duplicate file paths in manifest')
        result['small_files'] = [verify(entry, True) for entry in small]
        result['large_model'] = [verify(entry, args.deep_model) for entry in large]
        result['ok'] = all(row['ok'] for row in result['small_files']+result['large_model'])
    except Exception as error:
        result.update(ok=False, error=type(error).__name__ + ': ' + str(error))
    result['elapsed_seconds'] = time.monotonic() - start
    result['note'] = ('All listed file contents were hashed.' if args.deep_model else
                      'Small files were hashed. Large model checks cover existence and byte size only; use --deep-model for content verification.')
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    print(encoded, flush=True)
    if args.output:
        with args.output.open('x', encoding='utf-8') as target: target.write(encoded+'\n')
    return 0 if result['ok'] else 1


if __name__ == '__main__': raise SystemExit(main())
