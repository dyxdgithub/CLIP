# CLIP 项目工作指引

## 1. 项目定位与入口

- 本仓库提供 CLIP 图文编码模型的 Python 推理实现和预训练权重加载入口；使用示例与公开 API 说明见 [README.md](README.md)。
- 对外 Python API 是 `clip.available_models()`、`clip.load()`、`clip.tokenize()`；导出位置见 `clip/__init__.py:1` 和 `clip/clip.py:27`。
- 模型适用范围、偏差与限制见 [model-card.md](model-card.md)；涉及应用场景的改动应先阅读该文件。
- Git 跟踪的包目录名是小写 `clip/`。某些 Windows 工作区可能显示为 `CLIP/`；新增文件和引用一律使用小写路径，以免在区分大小写的系统上失效。

## 2. 目录导航

| 路径 | 职责 |
| --- | --- |
| `clip/clip.py` | 模型注册表、SHA-256 校验下载、图像预处理、JIT/普通模型加载和文本打包；见 `clip/clip.py:29-39, 43-87, 89-205`。 |
| `clip/model.py` | ModifiedResNet、VisionTransformer、文本 Transformer、CLIP 前向计算及从权重重建模型；见 `clip/model.py:94, 206, 243, 399`。 |
| `clip/simple_tokenizer.py` | 文本清理、字节级 BPE 编解码；见 `clip/simple_tokenizer.py:62-132`。 |
| `clip/bpe_simple_vocab_16e6.txt.gz` | 分词器词表资源；由 `clip/simple_tokenizer.py:11, 63-78` 加载。 |
| `hubconf.py` | 将模型名称转换成 PyTorch Hub 入口，并复用 `clip.load()`；见 `hubconf.py:1-40`。 |
| `tests/test_consistency.py` | 对每个已注册模型比较 CPU 上 JIT 与普通实现的输出；见 `tests/test_consistency.py:9-25`。 |
| `notebooks/` | 交互示例与 ImageNet 提示词实验，不是包运行时依赖。 |
| `data/` | 提示词和数据集获取说明；下载数据不是安装或测试的默认步骤。 |
| `setup.py`, `requirements.txt`, `MANIFEST.in` | 包依赖、开发用 `pytest` extra、BPE 词表打包配置。 |
| `.github/workflows/test.yml` | 现有 CI：Python 3.8 和指定的 PyTorch/torchvision CPU 组合，安装包后运行 `pytest`。 |

## 3. 核心调用链

1. `clip/__init__.py` 转出 `clip/clip.py` 的公开函数。
2. `clip.load(name, device, jit, download_root)` 查找注册名或本地 checkpoint；注册名对应的文件下载到缓存并校验 SHA-256，默认缓存目录为 `~/.cache/clip`（`clip/clip.py:43-72, 94-125`）。
3. 普通路径调用 `build_model(state_dict)`；JIT 路径修补设备引用及 CPU 精度，然后返回模型和与输入分辨率匹配的图像预处理（`clip/clip.py:127-202`）。
4. `clip.tokenize()` 使用 BPE 词表，添加起止标记，默认长度 77；超长文本默认报错，只有 `truncate=True` 才截断（`clip/clip.py:205-242`）。
5. `CLIP.forward()` 分别编码图像和文本，归一化后计算双向相似度 logits（`clip/model.py:340-372`）。

## 4. 开发与验证

- 先按 [README.md](README.md) 选择与机器 CPU/CUDA 环境匹配的 PyTorch 和 torchvision，再从仓库根目录安装：

  ```bash
  python -m pip install -e ".[dev]"
  ```

- 纯语法检查不需要下载模型：

  ```bash
  python -m compileall -q clip tests
  ```

- 完整现有测试命令与 CI 一致，可从仓库根目录运行：

  ```bash
  python -m pytest -q tests/test_consistency.py
  ```

- 上述测试按 `available_models()` 参数化，逐个在 CPU 加载 JIT 与普通模型；首次运行会下载全部对应权重，耗时并占用大量缓存。无网络或权重资源时，报告未运行原因，不要把语法检查说成模型一致性测试（`tests/test_consistency.py:9-25`）。
- 测试读取根目录的 `CLIP.png`，运行测试时保持仓库根目录为当前目录（`tests/test_consistency.py:15`）。

## 5. 修改约定

- 保持公开 API 的参数和返回值兼容；调整 `clip/clip.py` 时同步检查 README 示例、`hubconf.py` 和测试。
- 增加或调整模型名称时，同时检查 `_MODELS` 中的权重 URL、URL 内的 SHA-256、`available_models()`、Hub 入口和参数化测试（`clip/clip.py:29-39, 43-72, 89-91`; `hubconf.py:8-40`; `tests/test_consistency.py:9`）。
- 修改 `build_model()`、视觉编码器或精度转换时，同时考虑 ResNet/ViT、JIT/普通加载及 CPU/CUDA 路径；预处理分辨率从加载的模型中取得（`clip/model.py:243-296, 375-436`; `clip/clip.py:138-202`）。
- 修改文本处理时，维护 BPE 词表、起止标记、77 token 上下文和 `encode_text()` 取结束标记特征的约定（`clip/simple_tokenizer.py:63-78`; `clip/clip.py:205-242`; `clip/model.py:343-355`）。
- `clip/bpe_simple_vocab_16e6.txt.gz` 是运行时资源；不要遗漏 `MANIFEST.in` 中的打包条目。
- 遵循附近代码的风格，优先做局部修改；没有现成的格式化或静态检查配置，不要假设存在额外的 lint 命令。

## 6. 按任务定位

| 任务 | 首先检查 |
| --- | --- |
| 调整用户可见 API | `clip/__init__.py`、`clip/clip.py`、`README.md`、`hubconf.py` |
| 调整模型权重加载 | `clip/clip.py:29-72, 94-202`、`clip/model.py:399-436` |
| 调整图像输入 | `clip/clip.py:75-87`、`clip/model.py:94-155, 206-241` |
| 调整文本输入 | `clip/simple_tokenizer.py`、`clip/clip.py:205-242`、`clip/model.py:343-355` |
| 调整相似度输出 | `clip/model.py:358-372`、`tests/test_consistency.py` |
| 调整分发内容 | `setup.py`、`requirements.txt`、`MANIFEST.in`、`hubconf.py` |
| 调整实验或数据说明 | `notebooks/`、`data/`、`model-card.md` |

## 7. 提交前检查

1. 确认改动覆盖的公开调用链和两种加载路径；必要时增加针对性测试。
2. 运行适用的语法检查和测试，记录因权重下载或设备条件未执行的部分。
3. 核对新增路径在 Git 中的大小写，尤其是 `clip/` 与 `MANIFEST.in`。
4. 若行为、可用模型或适用范围变化，同步更新 README、模型卡或对应数据说明。
