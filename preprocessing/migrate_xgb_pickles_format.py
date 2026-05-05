"""
One-time migration utility for classifier pickle files.

Migrates old payload format to new format by:
1) adding top-level "xgboost_version" if missing
2) converting full_model_data["model"] -> full_model_data["model_ubj"]

The converted model is serialized via XGBoost UBJ memory format.
"""

import argparse
import pickle
import warnings
from pathlib import Path

import xgboost


DEFAULT_PATTERN = 'xgb_*_nparms*_std*_ignoreConf*.pkl'


def runtime_xgboost_version():
    return str(getattr(xgboost, '__version__', 'unknown'))


def serialize_xgb_model_to_ubj_buffer(model):
    booster = model.get_booster() if hasattr(model, 'get_booster') else model
    try:
        model_bytes = booster.save_raw(raw_format='ubj')
    except TypeError as exc:
        raise RuntimeError(
            'Current XGBoost version does not support save_raw(raw_format="ubj") for in-memory UBJ export.'
        ) from exc

    model_bytes = bytes(model_bytes)

    if not model_bytes:
        raise RuntimeError('Failed to serialize XGBoost model to UBJ memory buffer')

    return model_bytes


def load_pickle_safely(pkl_path):
    with open(pkl_path, 'rb') as fp:
        # Suppress XGBoost model-deserialization compatibility warning while loading legacy pickles.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'.*If you are loading a serialized model.*',
                category=Warning,
            )
            return pickle.load(fp)


def migrate_payload(payload, pkl_path):
    if not isinstance(payload, dict):
        raise TypeError(f'Payload is not a dict: {pkl_path}')

    changed = False

    if not payload.get('xgboost_version'):
        payload['xgboost_version'] = runtime_xgboost_version()
        changed = True

    full_model_data = payload.get('full_model_data')
    if not isinstance(full_model_data, dict):
        raise KeyError(f'Missing/invalid full_model_data in: {pkl_path}')

    model_ubj = full_model_data.get('model_ubj')
    if model_ubj:
        # Already migrated for model storage.
        return payload, changed

    model = full_model_data.get('model')
    if model is None:
        raise KeyError(
            f'Neither full_model_data["model_ubj"] nor full_model_data["model"] found in: {pkl_path}'
        )

    full_model_data['model_ubj'] = serialize_xgb_model_to_ubj_buffer(model)
    full_model_data.pop('model', None)
    changed = True

    return payload, changed


def resolve_output_path(src_path, output_dir, in_place):
    if in_place:
        return src_path

    if output_dir is None:
        output_dir = src_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f'{src_path.stem}_migrated{src_path.suffix}'


def migrate_pickles(input_dir, pattern, output_dir=None, in_place=False, dry_run=False):
    pkl_paths = sorted(Path(input_dir).glob(pattern))

    if not pkl_paths:
        print(f'No pickle files found in {input_dir} with pattern: {pattern}')
        return 0

    print(f'Found {len(pkl_paths)} pickle file(s) in {input_dir}')

    migrated = 0
    unchanged = 0
    failed = 0

    for pkl_path in pkl_paths:
        try:
            payload = load_pickle_safely(pkl_path)
            payload, changed = migrate_payload(payload, pkl_path)

            out_path = resolve_output_path(pkl_path, output_dir, in_place)

            if changed:
                if dry_run:
                    print(f'[DRY-RUN] Would migrate: {pkl_path} -> {out_path}')
                else:
                    with open(out_path, 'wb') as fp:
                        pickle.dump(payload, fp)
                    print(f'Migrated: {pkl_path} -> {out_path}')
                migrated += 1
            else:
                print(f'Unchanged (already migrated): {pkl_path}')
                unchanged += 1

        except Exception as exc:
            print(f'Failed: {pkl_path} ({exc})')
            failed += 1

    print('\nMigration summary:')
    print(f'  Migrated:  {migrated}')
    print(f'  Unchanged: {unchanged}')
    print(f'  Failed:    {failed}')

    return 0 if failed == 0 else 1


def parse_args():
    parser = argparse.ArgumentParser(
        description='Migrate classifier pickles to include xgboost_version and model_ubj buffer.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            'Usage:\n'
            '  python migrate_xgb_pickles_format.py /path/to/pickles --dry-run\n'
            '  python migrate_xgb_pickles_format.py /path/to/pickles --output-dir /path/to/out\n'
            '  python migrate_xgb_pickles_format.py /path/to/pickles --in-place\n'
        ),
    )
    parser.add_argument(
        'input_dir',
        type=Path,
        help='Folder containing classifier pickles',
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default=DEFAULT_PATTERN,
        help=f'Glob pattern to select pickle files (default: {DEFAULT_PATTERN})',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output folder for migrated pickles (default: same as input_dir)',
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Overwrite original pickle files instead of writing *_migrated.pkl files',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without writing files',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f'Input directory not found: {args.input_dir}')

    exit_code = migrate_pickles(
        input_dir=args.input_dir,
        pattern=args.pattern,
        output_dir=args.output_dir,
        in_place=args.in_place,
        dry_run=args.dry_run,
    )
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
