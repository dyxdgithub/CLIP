# CLIP 全参数微调基线

两个入口都更新 CLIP 图像编码器、文本编码器和相似度温度参数，默认使用单张 GPU。运行前在服务器安装本仓库依赖，并将两份 CSV 与图片放到服务器。图片根目录应满足
`<IMAGE_ROOT>/<RelativePath>`，例如
`<IMAGE_ROOT>/Aircraft/Fixed-wing aircraft/0000a90019e380dc.jpg`。
启动时会检查每个 ImageID 是否至少有一个对应图片；可用 `--skip-file-check` 跳过预检。
启动器默认设置 `ALLOW_TRUNCATED_IMAGES=1`：严格解码失败且 Pillow 明确报告图片截断时，
会重试读取并在终端警告中显示文件路径；这类图片的缺失像素可能影响数据质量。
需要严格拒绝截断图片时，设 `ALLOW_TRUNCATED_IMAGES=0`。该策略记录在运行的
`config.json` 中。
检索启动器默认设置 `SKIP_BAD_IMAGES=1`：无法解码的图片会被跳过，路径和错误
写入输出目录的 `bad_images.jsonl`，训练与验证指标会记录可用图像数和跳过数。
验证检索只使用成功读取的图像及其 caption。若某训练批次不足两张可用图像，
该批次不更新参数，并记录丢弃的单张图像数。设置 `SKIP_BAD_IMAGES=0` 可在坏图处报错。
正式比较模型时，应修复坏图或注明实际参与训练、验证和测试的样本数。
多标签分类启动器仍会在无法解码图片时报错并显示路径。
这份 8:1:1 划分来自原始 Open Images train 子集，报告结果时应称为自定义划分，
不要称为官方测试集。

训练日志需要 TensorBoard。在仓库根目录安装训练附加依赖：

```bash
python -m pip install -r my_code/train/requirements.txt
```

## 图文检索

```bash
bash my_code/train/run_finetune_clip_retrieval.sh
```

两个启动器默认从 `/media/xyycyc/Elements/dyx/TinyCLIP/my_code/data/OpenImage/meta/Hierarchy/n2/images_all` 读取图片；若图片另存于其他位置，可在命令前设置 `IMAGE_ROOT=/path/to/images_all`。

使用 `Image_IDs_n2_all_downloaded_split_8_1_1.csv` 的划分和
`Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv` 的 caption。
每张图片每轮只训练一次，多条 caption 按轮次轮换；同一批中完全相同的 caption
对应多张图时，采用多正例对比损失。验证和测试分别报告图搜文、文搜图的
Recall@1/5/10。验证选最佳 checkpoint，训练默认不评估测试集；
确定最终配置后再显式运行测试。
候选文本按完全相同的字符串去重，评估时允许一个查询有多个正确匹配。

## 多标签分类

```bash
bash my_code/train/run_finetune_clip_multilabel.sh
```

按 ImageID 汇总叶类别，计算图像与每个类别 prompt 的相似度，并对已知正负类别
分别平均二元交叉熵。验证和测试报告 macro/micro mAP；没有足够正负样本的类别
不计入 macro mAP，同时报告参与计算的类别数。

划分 CSV 只有正标签。默认把未列出的类别视为负例，输出的 mAP 是**弱标签代理指标**，
不宜与完整人工标注的 mAP 直接比较。如有 Open Images 的
`ImageID,Source,LabelName,Confidence` 人工验证标签 CSV，可传入
`--verified-labels-csv /path/to/labels.csv`；脚本会使用已验证负标签并屏蔽
其他未标注类别。请在两组实验中固定这个标签策略。

## 运行与恢复

启动器支持 `MODEL_NAME`、`DEVICE`、`PRECISION`、`BATCH_SIZE`、
`EPOCHS`、`NUM_WORKERS`、`LEARNING_RATE`、`SEED`、`OUTPUT_PATH`
环境变量，额外 Python 参数可直接附在启动命令末尾。默认模型为
`ViT-B/32`，若服务器没有缓存权重，`clip.load` 首次运行会下载。
可通过 `--download-root` 指定权重缓存，或以 `MODEL_NAME` 指向本地
预训练 CLIP state dict。

```bash
OUTPUT_PATH=/path/to/existing/run \
  bash my_code/train/run_finetune_clip_retrieval.sh \
  --resume /path/to/existing/run/last.pt
```

训练记录写入输出目录中的 `config.json`、`metrics.jsonl`、`tensorboard/`、
`last.pt` 和 `best.pt`。TensorBoard 保存 step 损失与学习率、epoch 损失、
验证损失和任务指标；执行测试时也保存测试指标。脚本按 epoch 保存断点；
从 `last.pt` 恢复时从下一轮继续，并在 TensorBoard 中延续全局 step。
检索启动器默认按时间和进程号为每次运行生成独立输出目录。若在首个 epoch
完成前失败，尚无 checkpoint；直接重跑会创建新目录。手动指定 `OUTPUT_PATH`
时，请为新运行指定新目录，保留旧目录中的诊断日志。恢复已有 checkpoint 时，
将 `OUTPUT_PATH` 指向该 checkpoint 所在目录。
`--eval-only` 的结果写入单独的 `tensorboard_eval/`，不会覆盖已有训练曲线。
在仓库根目录查看所有训练曲线：

```bash
tensorboard --logdir my_code/train/output
```
确定最终配置后，可用
`--eval-only --resume /path/to/best.pt` 评估验证集和测试集；
也可在最终训练命令中显式添加 `--test-after-train`。
单卡显存不足时可调小 `BATCH_SIZE`；检索任务的批大小也决定了每一步的
批内负例数，因此调整批大小须在实验记录中说明。
若在模型加载阶段就出现 CUDA out of memory，先运行 `nvidia-smi` 检查显存占用，
通过 `DEVICE=cuda:N` 选择有足够空闲显存的 GPU 或释放该卡上的其他任务；
此时训练尚未开始，调小 `BATCH_SIZE` 不能解决模型加载失败。
