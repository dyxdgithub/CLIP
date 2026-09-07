
import functools
import json
import logging
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
import io

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from open_clip.model import convert_to_new_checkpoint
from open_clip.factory import create_model_and_transforms, get_tokenizer
from open_clip import ClipLoss
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.optimizer import build_optimizer
from training.scheduler import cosine_lr
from training.my_meter import AverageMeter


# ============ 命令行参数定义 ============

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="TinyCLIP 全参数微调训练")

    # 必需参数
    parser.add_argument(
        "--model", type=str, required=True,
        help="模型名称，如 TinyCLIP_ViT_16")
    parser.add_argument(
        "--pretrained", type=str, required=True,
        help="预训练模型权重路径 (.pt 文件)")

    # 数据集输入（二选一）
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train-data", type=str,
        help="本地训练数据 CSV 文件路径")
    group.add_argument(
        "--hf-dataset", type=str,
        help="Hugging Face 数据集名称，如 wikimedia/wit_base")

    parser.add_argument(
        "--val-data", type=str, default=None,
        help="本地验证数据 CSV 文件路径")
    parser.add_argument(
        "--hf-val-dataset", type=str, default=None,
        help="Hugging Face 验证数据集名称")
    parser.add_argument(
        "--hf-split", type=str, default="train",
        help="Hugging Face 数据集使用的 split，默认 train")
    parser.add_argument(
        "--hf-image-col", type=str, default="image",
        help="Hugging Face 数据集中图像列名")
    parser.add_argument(
        "--hf-text-col", type=str, default="caption_reference_description",
        help="Hugging Face 数据集中文本列名")

    # 数据相关
    parser.add_argument(
        "--csv-separator", type=str, default="\t",
        help="CSV 文件分隔符，默认制表符")
    parser.add_argument(
        "--csv-img-key", type=str, default="filepath",
        help="CSV 中图像路径列名")
    parser.add_argument(
        "--csv-caption-key", type=str, default="title",
        help="CSV 中文本描述列名")
    parser.add_argument(
        "--train-num-samples", type=int, default=None,
        help="训练集样本数量（用于 webdataset）")

    # 训练相关
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="训练轮数")
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="每 GPU 批量大小")
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="学习率，默认 1e-5（较小适合微调）")
    parser.add_argument(
        "--wd", type=float, default=0.01,
        help="权重衰减")
    parser.add_argument(
        "--warmup", type=int, default=500,
        help="学习率预热步数")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="数据加载 worker 数量")
    parser.add_argument(
        "--gradient-clip", type=float, default=1.0,
        help="梯度裁剪阈值")

    # 精度相关
    parser.add_argument(
        "--precision", type=str, default="amp_bfloat16",
        choices=["amp", "amp_bfloat16", "fp16", "fp32"],
        help="训练精度")
    parser.add_argument(
        "--image-mean", type=float, nargs='+', default=None,
        help="图像归一化均值")
    parser.add_argument(
        "--image-std", type=float, nargs='+', default=None,
        help="图像归一化标准差")

    # 分布式训练
    parser.add_argument(
        "--dist-url", type=str, default="env://",
        help="分布式训练 URL")
    parser.add_argument(
        "--dist-backend", type=str, default="nccl",
        help="分布式后端")
    parser.add_argument(
        "--ddp-static-graph", action="store_true",
        help="启用 DDP 静态图优化")

    # 输出相关
    parser.add_argument(
        "--logs", type=str, default="./logs_finetune/",
        help="日志和检查点保存目录")
    parser.add_argument(
        "--name", type=str, default=None,
        help="实验名称，默认使用时间戳")
    parser.add_argument(
        "--save-frequency", type=int, default=1,
        help="每多少个 epoch 保存一次检查点")
    parser.add_argument(
        "--report-to", type=str, default="",
        help="日志报告目标：tensorboard, wandb, tensorboard,wandb")

    # 断点续训
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从检查点恢复训练")

    # 其他
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子")
    parser.add_argument(
        "--debug", action="store_true",
        help="开启调试模式，记录更多日志")
    parser.add_argument(
        "--eval-frequency", type=int, default=1,
        help="每多少个 epoch 进行一次验证评估")

    return parser.parse_args()


# ============ HuggingFace 数据集类 ============

class HuggingFaceDataset(Dataset):
    """
    用于加载 Hugging Face 数据集的 Dataset 类。

    支持直接从 Hugging Face 加载图像-文本对数据集，
    自动处理图像下载和预处理。
    """

    def __init__(self, dataset, image_col, text_col, preprocess_fn, tokenizer):
        self.dataset = dataset
        self.image_col = image_col
        self.text_col = text_col
        self.preprocess_fn = preprocess_fn
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 获取图像
        image = item[self.image_col]
        if isinstance(image, str):
            # 如果是 URL，从网络下载
            import requests
            response = requests.get(image, timeout=10)
            image = Image.open(io.BytesIO(response.content)).convert('RGB')
        elif hasattr(image, 'convert'):
            image = image.convert('RGB')

        # 预处理图像
        image = self.preprocess_fn(image)

        # 获取文本
        text = str(item[self.text_col])
        text = self.tokenizer(text)[0]

        return image, text


def get_hf_data(args, preprocess_fn, is_train, tokenizer, rank=0, world_size=1):
    """
    加载 Hugging Face 数据集。

    Args:
        args: 包含 hf_dataset, hf_split, hf_image_col, hf_text_col 等参数
        preprocess_fn: 图像预处理函数
        is_train: 是否为训练集
        tokenizer: 文本 tokenizer
        rank: 当前进程 rank
        world_size: 总进程数
    """
    if load_dataset is None:
        raise ImportError("需要安装 datasets 库: pip install datasets")

    hf_dataset_name = args.hf_dataset
    split = args.hf_split if is_train else (args.hf_val_split or "validation")
    image_col = args.hf_image_col
    text_col = args.hf_text_col

    logging.info(f"加载 Hugging Face 数据集: {hf_dataset_name} (split: {split})")

    # 加载数据集
    dataset = load_dataset(hf_dataset_name, split=split, trust_remote_code=True)

    # 过滤掉无效样本（缺少图像或文本的样本）
    def filter_valid(example):
        try:
            img = example[image_col]
            txt = example.get(text_col, None)
            if txt is None or str(txt).strip() == "":
                return False
            return True
        except Exception:
            return False

    # 仅在主进程进行过滤（避免重复日志）
    if rank == 0:
        logging.info(f"原始数据集大小: {len(dataset)}")
        original_size = len(dataset)

        # 检查样本格式
        sample = dataset[0]
        if image_col not in sample:
            raise ValueError(f"数据集缺少列 '{image_col}'，可用列: {list(sample.keys())}")
        if text_col not in sample:
            raise ValueError(f"数据集缺少列 '{text_col}'，可用列: {list(sample.keys())}")

        # 过滤无效样本
        dataset = dataset.filter(filter_valid, load_from_cache_file=False)
        logging.info(f"过滤后数据集大小: {len(dataset)} (移除了 {original_size - len(dataset)} 个无效样本)")

    # 广播过滤后的数据集大小（同步各进程）
    if world_size > 1:
        sizes = [len(dataset)]
        torch.distributed.all_reduce_object_list(sizes)
        if sizes[0] != len(dataset):
            # 重新加载过滤后的数据集
            dataset = load_dataset(hf_dataset_name, split=split, trust_remote_code=True)
            dataset = dataset.filter(filter_valid, load_from_cache_file=False)

    # 创建 Dataset
    hf_dataset = HuggingFaceDataset(dataset, image_col, text_col, preprocess_fn, tokenizer)

    # 创建 Sampler
    sampler = None
    shuffle = is_train
    if world_size > 1:
        sampler = torch.utils.data.DistributedSampler(
            hf_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=is_train
        )
        shuffle = False

    # 创建 DataLoader
    dataloader = DataLoader(
        hf_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=lambda x: (torch.stack([item[0] for item in x]),
                               torch.cat([item[1] for item in x]))
    )

    dataloader.num_samples = len(hf_dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader, sampler


# ============ 工具函数 ============

def random_seed(seed=42, rank=0):
    """设置随机种子"""
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_autocast(precision):
    """获取混合精度 autocast 上下文"""
    if precision == 'amp':
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16':
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif precision == 'fp32':
        return lambda: torch.cuda.amp.autocast(enabled=False)
    else:
        from contextlib import suppress
        return suppress


# ============ 训练函数 ============

def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args,
                    tb_writer=None, start_iter=0):

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()
    loss_fn = ClipLoss(
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=False)

    dataloader = data['train'].dataloader
    dataloader.device = args.device

    if start_iter == 0:
        data['train'].set_epoch(epoch)

    num_batches_per_epoch = dataloader.num_batches
    total_batch_size = args.batch_size * args.world_size
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i + start_iter

        # 学习率调度
        scheduler(step)

        # 数据加载
        images, texts = batch
        images = images.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)

        # 清零梯度
        for opt in optimizer:
            opt.zero_grad()

        # 前向传播
        with autocast():
            image_features, text_features, logit_scale = model(images, texts, normalized=True)
            loss = loss_fn(image_features, text_features, logit_scale)

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪
        if args.gradient_clip is not None:
            scaler.unscale_(optimizer[0])
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

        # 更新参数
        scaler.step(optimizer[0])
        scaler.update()

        # 限制 logit_scale 范围（原始论文限制在 ln(100) 以内）
        with torch.no_grad():
            model.logit_scale.clamp_(0, math.log(100))

        # 记录日志
        batch_time = time.time() - end
        loss_m.update(loss.item(), images.size(0))

        if is_master(args) and (i % 10 == 0 or i == num_batches_per_epoch - 1):
            percent = 100.0 * i / num_batches_per_epoch
            logging.info(
                f"Epoch: {epoch} [{i}/{num_batches_per_epoch}] "
                f"[{i * total_batch_size:>{sample_digits}}/{dataloader.num_samples} ({percent:.0f}%)] "
                f"Loss: {loss_m.val:.4f} ({loss_m.avg:.4f}) "
                f"LR: {optimizer[0].param_groups[0]['lr']:.2e} "
                f"Time: {batch_time:.3f}s"
            )

        # 记录到 tensorboard
        if tb_writer is not None and i % 50 == 0:
            tb_writer.add_scalar("train/loss", loss_m.val, step)
            tb_writer.add_scalar("train/lr", optimizer[0].param_groups[0]['lr'], step)
            tb_writer.add_scalar("train/batch_time", batch_time, step)

        end = time.time()

    return model, optimizer, scaler


def evaluate(model, data, epoch, args, tb_writer=None):
    """在验证集上评估模型"""
    from training.zero_shot import zero_shot_eval

    model.eval()
    metrics = zero_shot_eval(model, data, epoch, args)

    if not metrics:
        return {}

    if is_master(args):
        logging.info(f"Eval Epoch: {epoch} " + "\t".join([
            f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()
        ]))

        if tb_writer is not None:
            for name, val in metrics.items():
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        # 保存到 results.jsonl
        if args.checkpoint_path:
            results_file = os.path.join(args.checkpoint_path, "results.jsonl")
            with open(results_file, "a+") as f:
                f.write(json.dumps(metrics) + "\n")

    return metrics


# ============ 主函数 ============

def main():
    import time

    args = parse_args()

    # 设置随机种子
    random_seed(args.seed, 0)

    # 获取分布式环境信息
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    # 生成实验名称
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
        ])

    # 初始化日志
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_base_path = os.path.join(args.logs, args.name)
    os.makedirs(log_base_path, exist_ok=True)
    log_filename = f'out-{args.rank}.log' if args.log_local else 'out.log'
    args.log_path = os.path.join(log_base_path, log_filename)
    setup_logging(args.log_path, args.log_level)

    # 初始化分布式设备
    device = init_distributed_device(args)
    logging.info(f"初始化完成 | 设备: {args.device} | Rank: {args.rank}/{args.world_size}")

    # 保存检查点路径
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        os.makedirs(args.checkpoint_path, exist_ok=True)

    # 加载模型
    logging.info(f"加载模型: {args.model}")
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        pretrained='',
        precision=args.precision,
        device=device,
    )

    # 加载预训练权重
    logging.info(f"加载预训练权重: {args.pretrained}")
    state_dict = torch.load(args.pretrained, map_location='cpu')

    # 处理检查点格式
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        state_dict = state_dict['model']

    # 移除分布式包装的 module. 前缀
    if next(iter(state_dict.keys())).startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # 转换检查点格式
    state_dict = convert_to_new_checkpoint(state_dict)

    # 加载权重
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(f"缺少的 keys: {missing}")
    if unexpected:
        logging.warning(f"多余的 keys: {unexpected}")

    model.cuda()

    # 冻结图像编码器（可选，如果只微调文本编码器）
    # 如果需要全参数微调，注释掉下面两行
    # model.lock_image_tower()
    # logging.info("已锁定图像编码器，仅微调文本编码器")

    if is_master(args):
        # 打印模型参数量
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_params_img = sum(p.numel() for p in model.visual.parameters() if p.requires_grad)
        n_params_txt = sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
        logging.info(f"可训练参数量: {n_params / 1e6:.2f}M")
        logging.info(f"  - 图像编码器: {n_params_img / 1e6:.2f}M")
        logging.info(f"  - 文本编码器: {n_params_txt / 1e6:.2f}M")

    # 准备数据
    tokenizer = get_tokenizer(args.model)

    # 根据数据源选择数据加载方式
    if args.hf_dataset:
        # 使用 Hugging Face 数据集
        train_loader, train_sampler = get_hf_data(
            args, preprocess_train, is_train=True, tokenizer=tokenizer,
            rank=args.rank, world_size=args.world_size
        )
        data = {'train': type('DataInfo', (), {
            'dataloader': train_loader,
            'sampler': train_sampler,
            'set_epoch': lambda s, e: train_sampler.set_epoch(e) if train_sampler else None
        })()}

        # 验证集（如果有指定）
        if args.hf_val_dataset:
            val_loader, _ = get_hf_data(
                args, preprocess_val, is_train=False, tokenizer=tokenizer,
                rank=args.rank, world_size=args.world_size
            )
            data['val'] = type('DataInfo', (), {
                'dataloader': val_loader,
                'sampler': None
            })()
    else:
        # 使用本地 CSV 数据集
        data = get_data(
            args,
            (preprocess_train, preprocess_val),
            epoch=0,
            tokenizer=tokenizer
        )

    logging.info(f"数据集: {set(data.keys())}")

    # 创建优化器（所有参数）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        betas=(0.9, 0.999),
    )

    # 学习率调度器
    total_steps = data['train'].dataloader.num_batches * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # 混合精度
    use_loss_scale = args.precision in ['amp', 'fp16']
    scaler = GradScaler(enabled=use_loss_scale)

    # 断点续训
    start_epoch = 0
    start_iter = 0
    if args.resume is not None:
        logging.info(f"从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        logging.info(f"已恢复训练，从 epoch {start_epoch} 开始")

    # 分布式包装
    if args.world_size > 1:
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], **ddp_args)

    # TensorBoard
    writer = None
    if is_master(args) and 'tensorboard' in args.report_to:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(log_base_path, "tensorboard"))
        except ImportError:
            logging.warning("tensorboard 未安装，跳过")

    # 训练循环
    logging.info(f"开始训练 | Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f"========== Epoch {epoch} ==========")

        model, optimizer, scaler = train_one_epoch(
            model, data, epoch, optimizer, scaler, scheduler, args, writer, start_iter
        )
        start_iter = 0

        # 评估
        if (epoch + 1) % args.eval_frequency == 0 and 'val' in data:
            evaluate(model, data, epoch, args, writer)

        # 保存检查点
        if is_master(args) and (epoch + 1) % args.save_frequency == 0:
            ckpt_path = os.path.join(args.checkpoint_path, f"epoch_{epoch}.pt")
            save_dict = {
                "epoch": epoch,
                "state_dict": model.module.state_dict() if args.world_size > 1 else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
            }
            torch.save(save_dict, ckpt_path)
            logging.info(f"保存检查点到: {ckpt_path}")

    # 最终保存
    if is_master(args):
        final_path = os.path.join(args.checkpoint_path, "final.pt")
        save_dict = {
            "epoch": args.epochs - 1,
            "state_dict": model.module.state_dict() if args.world_size > 1 else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }
        torch.save(save_dict, final_path)
        logging.info(f"训练完成！最终模型保存到: {final_path}")

    logging.info("Done!")


if __name__ == "__main__":
    main()
