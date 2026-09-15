"""Create a reproducible split before preparing subsets or individual model branches."""
import argparse
from pathlib import Path
import pandas as pd
from panda.data import read_metadata, assign_folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='data/train.csv')
    parser.add_argument('--output', default='data/folds.csv')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude')
    parser.add_argument('--subset-per-grade', type=int, help='Optional stratified trial subset, sampled before splitting')
    args = parser.parse_args()
    frame = read_metadata(args.csv)
    if args.exclude:
        frame = frame[~frame.image_id.isin(pd.read_csv(args.exclude).image_id)]
    if args.subset_per_grade:
        frame = pd.concat([group.sample(min(args.subset_per_grade, len(group)), random_state=args.seed)
                           for _, group in frame.groupby('isup_grade')], ignore_index=True)
    frame = assign_folds(frame, args.folds, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f'Saved {len(frame)} rows to {output}')


if __name__ == '__main__':
    main()
