import argparse
import os
import torch
import torch.optim as optim
import config
from model import BertOneStageModel, BertTwoStageModel
from train import train_one_stage_model, train_two_stage_model
from dataloader import MavenLoader, DistributedMavenLoader
from evaluate import get_submission, evaluate_one_stage_model, evaluate_two_stage_model
from utils import load_checkpoint, is_one_stage, is_two_stage
from transformers import BertTokenizer


def main(args):
    if args.distributed:
        if not args.cuda:
            raise RuntimeError('"distributed" can only be set true when using cuda')
        # used in distributed training
        torch.cuda.set_device(args.local_rank)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '7899'
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    else:
        if args.local_rank != 0:
            raise RuntimeError('"local_rank" should be set to 0 if not using distributed training')
    
    tokenizer = BertTokenizer.from_pretrained(config.bert_pretrain_path)
    tokenizer.add_special_tokens({
        'additional_special_tokens': [config.trigger_start_token, config.trigger_end_token]
    })

    if is_one_stage(args.model):
        model = BertOneStageModel(pretrained_path=config.bert_pretrain_path)
        train = train_one_stage_model
    elif is_two_stage(args.model):
        model = BertTwoStageModel(pretrained_path=config.bert_pretrain_path)
        train = train_two_stage_model
    else:
        raise NotImplementedError()
    model.bert.resize_token_embeddings(len(tokenizer))
    if args.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), args.lr)
    if args.ckpt is not None:
        load_checkpoint(args.ckpt, model, optimizer)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.distributed:
        Loader = DistributedMavenLoader
    else:
        Loader = MavenLoader
    train_loader = Loader(args.maven_dir, tokenizer, 'train', args.batch_size)
    valid_loader = Loader(args.maven_dir, tokenizer, 'val', args.batch_size)
    test_loader  = Loader(args.maven_dir, tokenizer, 'test', args.batch_size)

    if args.task == 'train':
        train(args, model, optimizer, train_loader, valid_loader, epochs=args.epochs)
        result = get_submission(args, model, test_loader)
        with open(config.result_jsonl_path, 'w', encoding='utf-8') as f:
            f.write(result)
    elif args.task == 'test':
        if args.ckpt is None:
            raise RuntimeError('"ckpt" should be appointed when testing')
        if is_one_stage(args.model):
            evaluate_one_stage_model(args, model, valid_loader)
        elif is_two_stage(args.model):
            evaluate_two_stage_model(args, model, valid_loader)
        result = get_submission(args, model, test_loader)
        with open(config.result_jsonl_path, 'w', encoding='utf-8') as f:
            f.write(result)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyper-parameters settings
    parser.add_argument('--batch_size', default=8, type=int,
        help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
        help='Learning rate')
    parser.add_argument('--epochs', default=10, type=int,
        help='Training epochs')
    parser.add_argument('--grad_acum_step', default=1, type=int,
        help='Gradient accumulation step')

    parser.add_argument('--maven_dir', type=str,
        help='Path to MAVEN dataset directory')
    parser.add_argument('--task', choices=['train', 'test'],
        help='Running task')
    parser.add_argument('--model', choices=['bert-one-stage', 'bert-two-stage'],
        help='Model choice')

    parser.add_argument('--logging', type=int, default=0,
        help='Logging interval, 0 for no logging')
    parser.add_argument('--ckpt', type=str,
        help='Checkpoint path to load')

    parser.add_argument('--distributed', action='store_true', default=False,
        help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
        help='Local rank in distributed training')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    main(args)
