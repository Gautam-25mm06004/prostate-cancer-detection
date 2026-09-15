"""Train one contributor/fold; use --init for phase 2 or --resume after interruption."""
import argparse
from dataclasses import asdict
import math
import json
import hashlib
import random
import time
from pathlib import Path
import torch
from tqdm import tqdm
from panda.config import load_config
from panda.data import read_metadata, assign_folds, make_loader
from panda.engine import choose_device, seed_everything, to_device, save_json, save_checkpoint, validate
from panda.metrics import report
from panda.models import PandaModel, compute_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/drhb.json')
    parser.add_argument('--csv')
    parser.add_argument('--cache-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-tiles', type=int)
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--save-every', type=int, default=0, help='Also retain every Nth epoch for TTA selection (0 keeps best/last)')
    parser.add_argument('--checkpoint-every-updates', type=int, default=25, help='Save within an epoch every N optimizer updates (0 disables periodic saves)')
    parser.add_argument('--session-minutes', type=float, default=0, help='Stop at an optimizer boundary after this wall-time budget (0 has no budget)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--init', help='Load only model weights; begin a new training phase')
    group.add_argument('--resume', help='Resume model, optimizer, scheduler and RNG state')
    args = parser.parse_args()
    if args.checkpoint_every_updates < 0 or args.session_minutes < 0:
        parser.error('Checkpoint interval and session budget cannot be negative')
    session_start = time.monotonic()
    config = load_config(args.config)
    for key in ('csv', 'cache_dir', 'output_dir', 'epochs', 'fold', 'batch_size', 'num_tiles'):
        if getattr(args, key) is not None:
            setattr(config, key, getattr(args, key))
    if args.no_pretrained:
        config.pretrained = False
    config.validate()
    if config.num_workers != 0:
        raise ValueError('Use num_workers=0 for reproducible mid-epoch resumption')
    if not config.freeze_batchnorm and config.checkpoint_encoder:
        raise ValueError('Set freeze_batchnorm=true when checkpoint_encoder=true')
    seed_everything(config.seed)
    device = choose_device(args.device)
    frame = assign_folds(read_metadata(config.csv), config.folds, config.seed)
    if config.branch.startswith('rguo') and 'prediction_reg' not in frame:
        print('No teacher columns supplied: using supervised labels. Supply an OOF CSV to enable soft-target blending.')
    train = frame[frame.fold != config.fold].sort_values('image_id').reset_index(drop=True)
    valid = frame[frame.fold == config.fold].sort_values('image_id').reset_index(drop=True)
    metadata_hash = hashlib.sha256(frame.sort_values('image_id').to_csv(index=False).encode()).hexdigest()
    if train.empty or valid.empty:
        raise ValueError('Both training and validation folds must contain samples')
    if config.branch == 'rguo_high':
        provenance = json.loads((Path(config.cache_dir) / 'selector.json').read_text())
        if set(provenance['train_ids']) & set(valid.image_id):
            raise ValueError('Attention selector was trained on this validation fold; use a matching outer-fold selector/cache')
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists() and not args.resume:
        raise FileExistsError('This run already exists; use --resume or a new --output-dir')
    valid_loader = make_loader(valid, config)
    model = PandaModel(config, pretrained=False if args.init or args.resume else None).to(device)
    optimizer_class = torch.optim.RAdam if config.optimizer == 'radam' else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    updates = math.ceil(math.ceil(len(train) / config.batch_size) / config.accumulation_steps)
    if config.scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate,
                                                        total_steps=config.epochs*updates, pct_start=0.3)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs*updates)
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp and device.type == 'cuda')
    start, best = 0, -float('inf')
    history = []
    resume_samples, resume_total = 0, 0.0
    if args.init or args.resume:
        ckpt = torch.load(args.init or args.resume, map_location='cpu', weights_only=True)
        if ckpt.get('format_version') != 1:
            raise ValueError('Only modern project checkpoints are supported')
        cross_resolution_init = bool(args.init) and (ckpt['config']['branch'], config.branch) == ('rguo_mid_eff', 'rguo_high')
        if (ckpt['config']['branch'] != config.branch and not cross_resolution_init) or ckpt['config']['backbone'] != config.backbone:
            raise ValueError('Checkpoint model differs from requested branch/backbone')
        if set(ckpt['train_ids']) & set(valid.image_id):
            raise ValueError('Checkpoint training IDs overlap the current validation fold')
        model.load_state_dict(ckpt['model'])
        if args.resume:
            allowed_changes = {'csv', 'cache_dir', 'output_dir', 'num_workers'}
            for key, value in asdict(config).items():
                if key not in allowed_changes and value != ckpt['config'].get(key, getattr(type(config)(), key)):
                    raise ValueError(f'Cannot change {key} on resume; use --init for a new phase')
            if set(ckpt['train_ids']) != set(train.image_id) or set(ckpt['valid_ids']) != set(valid.image_id):
                raise ValueError('Resume requires the same train/validation IDs')
            if ckpt.get('metadata_sha256', metadata_hash) != metadata_hash:
                raise ValueError('Resume requires unchanged labels and metadata')
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            scaler.load_state_dict(ckpt['scaler'])
            complete = ckpt.get('epoch_complete', True)
            start, best, history = ckpt['epoch']+int(complete), ckpt['best'], ckpt['history']
            if not complete:
                resume_samples, resume_total = ckpt['next_sample'], ckpt['training_loss_sum']
                if ckpt.get('order_version') != 1 or not 0 <= resume_samples <= len(train):
                    raise ValueError('Unsupported or invalid within-epoch checkpoint')
            torch.set_rng_state(ckpt['rng_torch'])
            random.setstate(ckpt['rng_python'])
            if device.type == 'cuda' and ckpt.get('rng_cuda'):
                torch.cuda.set_rng_state_all(ckpt['rng_cuda'])
    # Preserve existing run records if checkpoint validation above rejects a resume.
    config.save(output / 'config.json')
    frame.to_csv(output / 'splits.csv', index=False)
    def snapshot(epoch, complete, total, samples):
        return {'format_version': 1, 'config': asdict(config), 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'epoch': epoch, 'epoch_complete': complete, 'next_sample': samples, 'training_loss_sum': total,
                'order_version': 1, 'metadata_sha256': metadata_hash,
                'best': best, 'history': history, 'train_ids': train.image_id.tolist(),
                'valid_ids': valid.image_id.tolist(), 'rng_torch': torch.get_rng_state(),
                'rng_python': random.getstate(), 'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}

    def session_expired():
        return args.session_minutes > 0 and time.monotonic()-session_start >= args.session_minutes*60

    print(f'{config.branch}: {len(train)} train / {len(valid)} validation, {device}, {config.num_tiles} tiles per slide')
    for epoch in range(start, config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, samples = (resume_total, resume_samples) if epoch == start else (0.0, 0)
        # A private generator fixes this epoch's order without consuming model RNG state.
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(config.seed+epoch)).tolist()
        train_loader = make_loader(train.iloc[order].iloc[samples:], config, True, ordered=True)
        update_count = math.ceil(samples / (config.batch_size * config.accumulation_steps))
        # Accumulate summed per-slide losses, then divide by the actual group size.
        # This also handles a short final batch and a partial accumulation group.
        group_samples = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')):
            batch = to_device(batch, device)
            size = len(batch['label'])
            with torch.amp.autocast(device.type, enabled=config.amp and device.type == 'cuda'):
                loss = compute_loss(model(batch['tiles']), batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss; inspect data and learning rate')
            scaler.scale(loss * size).backward()
            group_samples += size
            total += float(loss.detach()) * size
            samples += size
            if (step+1) % config.accumulation_steps == 0 or step+1 == len(train_loader):
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_samples)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                group_samples = 0
                update_count += 1
                stop = session_expired()
                if stop or samples == len(train) or (args.checkpoint_every_updates and update_count % args.checkpoint_every_updates == 0):
                    save_checkpoint(snapshot(epoch, False, total, samples), output / 'last.pt')
                if stop:
                    print(f'Session budget reached. Resume from {output / "last.pt"}; {samples}/{len(train)} training slides completed this epoch.')
                    return
        val_loss, predictions = validate(model, valid_loader, device)
        # Remove old teacher columns before merging current predictions.
        clean_valid = valid.drop(columns=['score', 'prediction_reg', 'classification'] + [f'prob_{i}' for i in range(6)], errors='ignore')
        evaluated = clean_valid.merge(predictions, on='image_id', validate='one_to_one')
        metrics = report(evaluated)
        rank = metrics['qwk'] if metrics['qwk'] is not None else -val_loss
        improved = rank > best
        best = max(best, rank)
        history.append({'epoch': epoch+1, 'train_loss': total/samples, 'valid_loss': val_loss, **metrics})
        ckpt = snapshot(epoch, True, total, samples)
        if args.save_every > 0 and (epoch+1) % args.save_every == 0:
            save_checkpoint(ckpt, output / f'epoch_{epoch+1:03d}.pt')
        if improved:
            save_checkpoint(ckpt, output / 'best.pt')
            evaluated.to_csv(output / 'validation_predictions.csv', index=False)
            save_json(metrics, output / 'metrics.json')
        save_json(history, output / 'history.json')
        # Publish completed-epoch state after the best model and result artifacts.
        save_checkpoint(ckpt, output / 'last.pt')
        print(f'Train loss {total/samples:.4f}, validation loss {val_loss:.4f}, QWK {metrics["qwk"]}')
        if session_expired():
            print(f'Session budget reached. Resume from {output / "last.pt"}.')
            return
    print(f'Run saved to {output}')


if __name__ == '__main__':
    main()
