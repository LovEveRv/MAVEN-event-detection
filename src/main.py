import argparse
import torch
import torch.optim as optim
import config
from model import BertOneStageModel, BertTwoStageModel
from train import train_one_stage_model, train_two_stage_model
from dataloader import MavenLoader
from transformers import BertTokenizer


def main(args):
    tokenizer = BertTokenizer.from_pretrained(config.bert_pretrain_path)
    tokenizer.add_special_tokens({
        'additional_special_tokens': [config.trigger_start_token, config.trigger_end_token]
    })

    if args.model == 'bert-one-stage':
        model = BertOneStageModel(pretrained_path=config.bert_pretrain_path)
        train = train_one_stage_model
    elif args.model == 'bert-two-stage':
        model = BertTwoStageModel(pretrained_path=config.bert_pretrain_path)
        train = train_two_stage_model
    else:
        raise NotImplementedError()
    model.bert.resize_token_embeddings(len(tokenizer))
    if args.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), args.lr)

    train_loader = MavenLoader(args.maven_dir, tokenizer, 'train', args.batch_size)
    valid_loader = MavenLoader(args.maven_dir, tokenizer, 'val', args.batch_size)
    test_loader  = MavenLoader(args.maven_dir, tokenizer, 'test', args.batch_size)

    if args.task == 'train':
        train(args, model, optimizer, train_loader, valid_loader, epochs=args.epochs)


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
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    main(args)
