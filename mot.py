import argparse
from pathlib import Path
from motlib.utils import get_args_parser
from motlib.engine.task_engine import build_engin
import torch.distributed as dist
from motlib.utils import task_setup


def main(args):
    task_setup(args)
    trainer = build_engin(args.train_engine, args)
    if not args.eval:
        trainer.train()
    else:
        trainer.test()
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ColTrack evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
