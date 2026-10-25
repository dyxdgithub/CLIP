# 下载三个 Open Images captioned_only CSV 的图片

从仓库根目录运行：

```bash
python my_code/data/scripts/download__images.py
```

脚本读取 `my_code/data/OpenImages/` 下的 train、test、validation 三份
`Image IDs_with_captions_and_labels_captioned_only_<split>.csv`，按
`ImageID` 去重，分别保存到 `OpenImages/train/`、`OpenImages/test/`
和 `OpenImages/val/`。图片统一以 `ImageID.jpg` 命名；原始 URL 返回
PNG 等格式时，验证后转换为 JPEG。已有且已验证的图片会复用。

每张图片依次尝试 [Open Images 官方说明](https://storage.googleapis.com/openimages/web/download_v7.html)
所链接的 [CVDF 公共 S3 镜像](https://github.com/cvdfoundation/open-images-dataset)
中的 `<split>/<ImageID>.jpg`、CSV 的 `OriginalURL`、CSV 的
`Thumbnail300KURL`。S3 镜像主要覆盖密集标注子集，因此不在镜像里的 ID
会转向 CSV 原图 URL。下载失败会按来源重试，然后尝试下一个来源。HTTP 404/410
会记入检查点，后续运行跳过该失效 URL。

默认同时下载 64 张图片，可根据网络和服务器限制调整：

```bash
python my_code/data/scripts/download__images.py --workers 32 --timeout 45 --retries 5
```

首次启动会读取三份 CSV 并建立去重索引；之后 CSV 文件未变时直接使用 SQLite
检查点。普通续跑直接跳过数据库中 `ok` 的图片，只检查未完成图片并重试失败项，
不会再对每张已完成图片执行文件查找。检查点默认为
`OpenImages/.openimages_v7_download.sqlite3`，最新计数在
`OpenImages/download_summary.json`。如果在脚本运行之外删除或修改过图片，先用
`--check-existing` 重新检查所有已完成图片的文件大小；用 `--verify-existing`
重新解码所有图片。`--retry-permanent` 会重试此前返回 404/410 的 URL。

逐图片状态 CSV 默认不再每次重写；需要导出时加 `--write-status-csv`，输出
`captioned_only_download_status_<split>.csv`。已有的状态 CSV 不会自动刷新，
应以 SQLite 检查点和 `download_summary.json` 为准。CSV 按唯一 ImageID 记录，
不重复列出同一图片的多条 caption。

如果图片目录在慢速外接盘上，可将 SQLite 检查点及报告放在更快的本地盘，
例如服务器上的 ext4：

```bash
python my_code/data/scripts/download__images.py \
  --state-dir /home/xyycyc/openimages-download-state
```

第一次指定新的 `--state-dir` 时，脚本会用 SQLite 备份 API 复制旧检查点
（包括尚未合并的 WAL 数据），不重新下载已有图片。迁移期间不要同时运行旧版
下载进程。原检查点保留在图片目录中，后续运行保持使用同一个 `--state-dir`。
`--state-dir` 只改变检查点和报告位置，图片仍写入 `--output-dir`。

可先小规模检查：

```bash
python my_code/data/scripts/download__images.py --split train --max-image-ids 100
```

完整运行只有在所有请求的 ImageID 都有有效图片时返回 0。第三方原图 URL
可能失效，官方镜像也不覆盖全部图片；无法取得的图片会保持 `failed`，
脚本返回非零，并在状态 CSV 中写明尝试结果。不能用网络重试保证已经下线的
源图片一定可得。
