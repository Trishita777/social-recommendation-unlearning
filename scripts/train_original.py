"""Train an original backbone without invoking historical deletion experiments."""
import argparse
from types import SimpleNamespace


def main(backbone=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if backbone is None:
        parser.add_argument('--backbone', choices=['ODSR', 'EGCL'], required=True)
    else:
        parser.set_defaults(backbone=backbone)
    parser.add_argument('--dataset', choices=['epinions', 'yelp2018'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, help='Override only for pilots; omit for the frozen recipe')
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.backbone == 'ODSR':
        from run_unlearning import run
        run(SimpleNamespace(dataset=args.dataset, seed=args.seed, device=args.device,
                            original_epochs=args.epochs or 30, unlearn_epochs=5,
                            batch_size=2048, strategy='random', fractions=[], methods=[]))
    else:
        from run_egcl_backbone import run
        run(SimpleNamespace(dataset=args.dataset, seed=args.seed, device=args.device,
                            method='original', fraction=.01, epochs=args.epochs or 2000,
                            pilot=False))


if __name__ == '__main__':
    main()
