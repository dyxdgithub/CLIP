# 测试检索微调模型

`test_clip_retrieval.py` 加载 `finetune_clip_retrieval.py` 保存的 `best.pt` 或
`last.pt`，自动读取同目录的 `config.json`。默认评估自定义划分的 test 集。
使用验证集选出的 `best.pt` 进行最终测试；不要根据测试指标反复选择 checkpoint。

从项目根目录运行：

```bash
bash my_code/test/run_test_clip_retrieval.sh /path/to/training_run/best.pt
```

服务器项目根目录为 `/media/xyycyc/Elements1/dyx/CLIP`。启动器默认使用训练时的
服务器图片根目录及 `cuda:3`。可通过 `IMAGE_ROOT`、`DEVICE`、`PRECISION`、
`EVAL_SPLIT`、`EVAL_BATCH_SIZE`、`SCORE_BATCH_SIZE`、`NUM_WORKERS`、
`PYTHON_BIN` 环境变量覆盖，或在命令末尾添加 Python 参数。

```bash
DEVICE=cpu bash my_code/test/run_test_clip_retrieval.sh /path/to/best.pt \
  --split-csv my_code/data/Image_IDs_n2_all_downloaded_split_8_1_1.csv \
  --caption-csv "my_code/data/Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv"
```

也可直接运行 Python，默认继承训练 config 中的批大小、图片处理和随机种子：

```bash
python my_code/test/test_clip_retrieval.py --checkpoint /path/to/best.pt --device cuda:3
```

`--split validation` 只评估验证集，`--split both` 依次评估验证集和测试集。
命令行参数优先于训练 config。如果没有 config，需要指定 `--image-root`，
必要时指定两份 CSV。CSV 内容通过 checkpoint 中的 SHA-256 校验，文件可以移动，
内容须与训练一致。模型直接从 checkpoint 重建，不下载初始预训练权重。

默认输出为 `my_code/test/output/clip_retrieval_<训练目录>_<checkpoint>_<时间>/`，
可用 `--output-dir` 指定空目录。保存 `config.json`、`metrics.json`、
`metrics.jsonl` 和 `tensorboard/`；启用 `--skip-bad-images` 时还会记录
`bad_images.jsonl`。坏图、截断图策略默认继承训练 config，可显式使用
`--skip-bad-images` / `--no-skip-bad-images` 以及
`--allow-truncated-images` / `--no-allow-truncated-images` 覆盖。

指标复用训练实现：图搜文与文搜图的 R@1/5/10、六项召回率均值、批内对比损失、
可用图像数及跳过的坏图数。召回率使用所选 split 的全部有效候选，重复 caption
作为多个正例处理。损失依赖评估批大小及采样 seed，比较损失时应保持二者一致。
该脚本评估图文检索，不包含类别 prompt 分类评估。

```bash
tensorboard --logdir my_code/test/output
```

## Flickr30K：微调前后对比

原始 OpenAI CLIP `ViT-B/32` 可用同一套 Flickr30K 检索评估直接测试，
不需要先把它保存成微调 checkpoint。建议传入目标训练运行的 `config.json`，
以复用其固定的数据集 revision 并核对测试集摘要：

```bash
bash my_code/test/run_test_clip_flickr30k.sh pretrained ViT-B/32 \
  --config my_code/train/output/<run>/config.json
```

测试微调模型则传入该运行的 `best.pt`：

```bash
bash my_code/test/run_test_clip_flickr30k.sh my_code/train/output/<run>/best.pt
```

两种模式默认都评估 test 集，使用相同的图文检索 Recall@1/5/10 和批内损失，
并将指标及 TensorBoard 日志写入 `my_code/test/output/` 下各自的新目录。
如要比较批内损失，使用相同的 `EVAL_BATCH_SIZE`；Recall 指标按整个
test 集计算。`EVAL_SPLIT=validation` 可用于诊断，最终报告使用 test 集。

环境依赖与训练一致：`python -m pip install -r my_code/train/requirements.txt`。
