"""
@Time   :   2021-01-12 15:23:56
@File   :   main.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import argparse
import os
import torch
from transformers import BertTokenizer
import pytorch_lightning as pl
from src.dataset import get_corrector_loader
from src.models import SoftMaskedBertModel
from src.data_processor import preproc
from src.utils import get_abs_path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_device", default='cpu', type=str, help="硬件，cpu or cuda")
    parser.add_argument("--gpu_index", default=0, type=int, help='gpu索引, one of [0,1,2,3,...]')
    parser.add_argument("--load_checkpoint", nargs='?', const=True, default=False, type=str2bool,
                        help="是否加载训练保存的权重, one of [t,f]")
    parser.add_argument('--bert_checkpoint', default='bert-base-chinese', type=str)
    parser.add_argument('--model_save_path', default='checkpoint', type=str)
    parser.add_argument('--epochs', default=10, type=int, help='训练轮数')
    parser.add_argument('--batch_size', default=8, type=int, help='批大小')
    parser.add_argument('--warmup_epochs', default=6, type=int, help='warmup轮数, 需小于训练轮数')
    parser.add_argument('--lr', default=1e-4, type=float, help='学习率')
    parser.add_argument('--accumulate_grad_batches',
                        default=16,
                        type=int,
                        help='梯度累加的batch数')
    parser.add_argument('--mode', default='train', type=str,
                        help='代码运行模式，以此来控制训练测试或数据预处理，one of [train, test, preproc]')
    parser.add_argument('--loss_weight', default=0.8, type=float, help='论文中的lambda，即correction loss的权重')
    arguments = parser.parse_args()
    if arguments.hard_device == 'cpu':
        arguments.device = torch.device(arguments.hard_device)
    else:
        arguments.device = torch.device(f'cuda:{arguments.gpu_index}')
    if not 0 <= arguments.loss_weight <= 1:
        raise ValueError(f"The loss weight must be in [0, 1], but get{arguments.loss_weight}")
    print(arguments)
    return arguments


def main():
    args = parse_args()
    if args.mode == 'preproc':
        print('preprocessing...')
        preproc()
        return

    tokenizer = BertTokenizer.from_pretrained(args.bert_checkpoint)
    model = SoftMaskedBertModel(args, tokenizer)
    train_loader = get_corrector_loader(get_abs_path('data', 'train.json'),
                                        tokenizer,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=4)
    valid_loader = get_corrector_loader(get_abs_path('data', 'dev.json'),
                                        tokenizer,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=4)
    test_loader = get_corrector_loader(get_abs_path('data', 'test.json'),
                                       tokenizer,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=4)
    trainer = pl.Trainer(max_epochs=args.epochs,
                         gpus=None if args.hard_device == 'cpu' else [args.gpu_index],
                         accumulate_grad_batches=args.accumulate_grad_batches)
    model.load_from_transformers_state_dict(get_abs_path('checkpoint', 'pytorch_model.bin'))
    if args.load_checkpoint:
        model.load_state_dict(torch.load(get_abs_path('checkpoint', f'{model.__class__.__name__}_model.bin')))
    if args.mode == 'train':
        trainer.fit(model, train_loader, valid_loader)

    model.load_state_dict(torch.load(get_abs_path('checkpoint', f'{model.__class__.__name__}_model.bin')))
    trainer.test(model, test_loader)


if __name__ == '__main__':
    main()
