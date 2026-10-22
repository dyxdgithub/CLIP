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
检查点。中断后运行同一命令即可续跑。检查点为
`OpenImages/.openimages_v7_download.sqlite3`；每个 split 的状态在
`captioned_only_download_status_<split>.csv`，总数在
`download_summary.json`。状态文件按唯一 ImageID 记录，不重复列出同一图片的
多条 caption。每次运行会重试仍失败的图片；用 `--retry-permanent`
也可重试此前返回 404/410 的 URL。用 `--verify-existing` 可重新解码全部已有图片。

可先小规模检查：

```bash
python my_code/data/scripts/download__images.py --split train --max-image-ids 100
```

完整运行只有在所有请求的 ImageID 都有有效图片时返回 0。第三方原图 URL
可能失效，官方镜像也不覆盖全部图片；无法取得的图片会保持 `failed`，
脚本返回非零，并在状态 CSV 中写明尝试结果。不能用网络重试保证已经下线的
源图片一定可得。
