
import argparse
from pathlib import Path
import shutil
import subprocess
import zipfile
import pandas as pd


def download_file(cli, image_id, folder, destination):
    if Path(image_id).name != image_id or '\\' in image_id:
        raise ValueError('Invalid image_id')
    filename = image_id + ('_mask.tiff' if folder == 'train_label_masks' else '.tiff')
    target = destination / folder / filename
    if target.is_file() and target.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = destination / '.downloads' / folder / image_id
    staging.mkdir(parents=True, exist_ok=True)
    subprocess.run([cli, 'competitions', 'download', 'prostate-cancer-grade-assessment',
                    '-f', f'{folder}/{filename}', '-p', str(staging)], check=True)
    raw = list(staging.rglob(filename))
    temporary = target.with_suffix('.partial')
    if raw:
        with raw[0].open('rb') as source, temporary.open('wb') as output:
            shutil.copyfileobj(source, output, length=1024*1024)
        temporary.replace(target)
        raw[0].unlink()
        return
    for archive_path in staging.rglob('*.zip'):
        with zipfile.ZipFile(archive_path) as archive:
            matches = [name for name in archive.namelist() if Path(name).name == filename]
            if len(matches) != 1:
                continue
            # Stream the requested member into a fixed destination; never extract arbitrary paths.
            with archive.open(matches[0]) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output, length=1024*1024)
        temporary.replace(target)
        archive_path.unlink()
        return
    raise FileNotFoundError(f'Kaggle did not return {folder}/{filename}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='data/folds.csv')
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--limit', type=int, default=60, help='Maximum slides; default 60, never the whole competition automatically')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    cli = shutil.which('kaggle')
    if cli is None:
        raise RuntimeError('Install the Kaggle CLI with pip install kaggle and authenticate before downloading')
    if args.test and args.masks:
        parser.error('Competition test label masks are not available')
    frame = pd.read_csv(args.csv, dtype={'image_id': str}).head(args.limit)
    if frame.empty or frame.image_id.duplicated().any():
        raise ValueError('CSV must contain unique image IDs')
    destination = Path(args.output_dir)
    for index, image_id in enumerate(frame.image_id):
        print(f'Downloading {index+1}/{len(frame)}: {image_id}', flush=True)
        download_file(cli, image_id, 'test_images' if args.test else 'train_images', destination)
        if args.masks:
            download_file(cli, image_id, 'train_label_masks', destination)
    frame.to_csv(destination / 'downloaded.csv', index=False)
    print(f'Downloaded only {len(frame)} slides. Metadata: {destination / "downloaded.csv"}')


if __name__ == '__main__':
    main()
