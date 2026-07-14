# Protein AI Web

本地运行的蛋白质 AI 预测网站，暗黑科幻风界面，推理完全在本地执行。

## 功能

- **二级结构预测** — 残基级 H/E/C 分类（V1 + ESM-2 集成，Q3 89.8%）
- **EC 酶分类** — 七大类酶功能识别（ESM-2 + MLP，87.3%）
- **突变效应预测** — 零样本致病性打分（ESM-2 LLR）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# ESM-2 模型会在首次运行时自动从 HuggingFace 下载
# 或者手动放到 inference/esm_model/ 目录

# 启动服务器
python server.py
```

打开 http://localhost:8765

## 项目结构

```
protein-ai-web/
├── server.py          # HTTP 服务器 + API 路由
├── api_backend.py     # ML 推理封装 + HTML/图表生成
├── index.html         # 前端界面（React/JSX）
├── start.bat          # Windows 一键启动
├── static/            # 运行时生成的图表图片
└── inference/
    ├── inference.py          # 二级结构预测
    ├── ec_inference.py       # EC 酶分类
    ├── mutation_inference.py # 突变效应预测
    └── ec_classifier.pt      # EC 分类器权重
```

## 技术栈

- 后端：Python http.server + 直接调用 ML 推理
- 前端：React (CDN) + Tailwind CSS
- ML：ESM-2 (facebook/esm2_t6_8M_UR50D)
- 不依赖 Gradio 运行时
