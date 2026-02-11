# arXiv CV Papers 自动化系统 - 项目总结

## 📋 项目概述

本项目是一个完全自动化的 arXiv 论文抓取和分析系统，专注于计算机视觉领域的视频生成（Video Generation）和视频编辑（Video Editing）相关论文。

**核心功能**：
- ✅ 每天自动从 arXiv 抓取最新论文
- ✅ 智能筛选视频相关研究
- ✅ 自动去重（与最近 5 天对比）
- ✅ 中英文双语支持
- ✅ 提取项目和代码链接
- ✅ 两个版本可选：V1（基础）和 V2（AI 深度分析）

## 🏗️ 系统架构

```
arxiv-cv-papers/
├── .github/
│   └── workflows/
│       └── daily-update.yml      # GitHub Actions 工作流
├── scripts/
│   ├── fetch_papers_v1.py        # V1 基础版脚本
│   └── fetch_papers_v2.py        # V2 增强版脚本（AI 分析）
├── papers/                       # 论文存储目录（自动生成）
│   ├── 2026-02-11.md
│   ├── 2026-02-10.md
│   └── ...
├── README.md                     # 项目主页
├── SETUP.md                      # 配置指南
├── USAGE.md                      # 使用指南
├── deploy.sh                     # 部署脚本
└── .gitignore                    # Git 忽略文件
```

## 🔧 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 数据源 | arXiv API | 免费、开放的学术论文数据库 |
| 自动化 | GitHub Actions | 定时任务调度和执行 |
| 编程语言 | Python 3.11 | 数据处理和分析 |
| AI 模型 | Gemini 2.5 Flash | 翻译和论文分析（免费） |
| PDF 处理 | poppler-utils | PDF 文本提取 |
| 存储 | GitHub Repository | 版本控制和数据存储 |

## 📊 两个版本对比

### V1 基础版

**特点**：
- 快速运行（2-5 分钟）
- 低 API 调用成本
- 适合日常使用

**包含内容**：
- 论文标题（中英文）
- 论文摘要（中英文）
- 作者列表
- 发布日期
- arXiv ID 和 PDF 链接
- 项目主页和代码仓库链接
- 论文类别

### V2 增强版

**特点**：
- 深度分析（10-30 分钟）
- 中等 API 调用成本
- 适合深度研究

**额外包含**：
- PDF 全文下载和分析
- AI 阅读分析，包括：
  - 核心观点（简明扼要）
  - 技术方法概述
  - 实验验证评估
  - 结果可靠性分析
  - 实用价值评估
  - 批判性评价

## 🎯 核心功能详解

### 1. 多类别覆盖

抓取以下 arXiv 类别的论文：
- `cs.CV` - Computer Vision and Pattern Recognition
- `cs.AI` - Artificial Intelligence
- `cs.MM` - Multimedia
- `cs.RO` - Robotics
- `cs.LG` - Machine Learning

### 2. 智能筛选

使用关键词匹配筛选视频相关论文：
- video generation / synthesis
- video editing / edit
- video diffusion
- text-to-video / image-to-video
- video understanding / model
- temporal / motion generation
- video quality / enhancement / restoration

### 3. 自动去重

- 检查最近 3 天的新论文
- 与最近 5 天的历史记录对比
- 完全重复的论文不会重复收录

### 4. 链接提取

自动从论文摘要中提取：
- GitHub/GitLab 代码仓库链接
- 项目主页链接
- 其他相关资源链接

### 5. 中文翻译

使用 Gemini 2.5 Flash 模型：
- 翻译论文标题
- 翻译论文摘要
- 保留专业术语的英文形式

### 6. AI 深度分析（V2）

- 下载完整 PDF
- 提取全文内容
- AI 批判性阅读
- 生成简明分析报告

## ⚙️ 配置说明

### 必需配置

| 配置项 | 位置 | 说明 |
|--------|------|------|
| `OPENAI_API_KEY` | GitHub Secrets | 用于调用 Gemini 模型 |

### 可选配置

| 配置项 | 位置 | 默认值 | 说明 |
|--------|------|--------|------|
| `VERSION` | GitHub Variables | `v1` | 选择运行版本 |
| Cron 表达式 | workflow 文件 | `0 2 * * *` | 每天 UTC 02:00（北京时间 10:00） |
| `DAYS_TO_CHECK` | Python 脚本 | `3` | 检查最近 N 天的论文 |
| `DAYS_TO_COMPARE` | Python 脚本 | `5` | 与最近 N 天对比去重 |

## 🚀 部署流程

### 方式 1：使用部署脚本（推荐）

```bash
cd /home/ubuntu/arxiv-cv-papers
./deploy.sh
```

按提示输入 GitHub 用户名，脚本会自动完成推送。

### 方式 2：手动部署

```bash
cd /home/ubuntu/arxiv-cv-papers

# 设置分支
git branch -M main

# 添加远程仓库（替换 YOUR_USERNAME）
git remote add origin https://github.com/YOUR_USERNAME/arxiv-cv-papers.git

# 推送代码
git push -u origin main
```

### 后续步骤

1. **在 GitHub 上创建仓库**（如果还没有）
   - 访问 https://github.com/new
   - 仓库名：`arxiv-cv-papers`

2. **配置 Secrets**
   - Settings → Secrets and variables → Actions → Secrets
   - 添加 `OPENAI_API_KEY`

3. **配置版本（可选）**
   - Settings → Secrets and variables → Actions → Variables
   - 添加 `VERSION` = `v1` 或 `v2`

4. **启用 Actions**
   - Actions → Daily arXiv Papers Update
   - Run workflow

## 📈 运行机制

### 定时任务

- **触发时间**：每天 UTC 02:00（北京时间 10:00）
- **执行环境**：GitHub Actions（Ubuntu latest）
- **超时限制**：6 小时（免费版）

### 工作流程

1. **Checkout 代码**：拉取最新代码
2. **安装依赖**：Python 包和系统工具
3. **确定版本**：读取 `VERSION` 变量
4. **运行脚本**：执行 V1 或 V2 脚本
5. **提交更新**：将新论文提交到仓库
6. **推送更改**：自动推送到 GitHub

### 错误处理

- **arXiv API 限流**：自动重试（最多 3 次），每次间隔 5 秒
- **请求间隔**：每个类别之间间隔 3 秒
- **超时保护**：单次 API 请求超时 30 秒
- **无新论文**：不会创建空文件或提交

## 🔍 输出格式

### Markdown 文件结构

```markdown
# arXiv Papers - 2026-02-11

**更新时间**: 2026-02-11 10:05:23
**论文数量**: 5
**版本**: V1 / V2

---

## 1. Paper Title

**中文标题**: 论文标题

**作者**: Author1, Author2, ...

**发布日期**: 2026-02-10

**arXiv ID**: [2402.xxxxx](https://arxiv.org/abs/2402.xxxxx)

**PDF**: [下载链接](https://arxiv.org/pdf/2402.xxxxx)

**项目主页**: https://project-url.com

**代码仓库**: https://github.com/user/repo

**类别**: cs.CV, cs.AI

<details>
<summary><b>摘要 (Abstract)</b></summary>

English abstract text...

</details>

<details>
<summary><b>中文摘要</b></summary>

中文摘要内容...

</details>

<!-- V2 版本额外包含 -->
<details>
<summary><b>🤖 AI 阅读分析</b></summary>

**核心观点**: ...
**技术方法**: ...
**实验验证**: ...
**结果可靠性**: ...
**实用价值**: ...
**批判性评价**: ...

</details>

---
```

### README 索引

```markdown
## 📅 论文列表

- [2026-02-11](papers/2026-02-11.md) - 5 篇论文 (V1)
- [2026-02-10](papers/2026-02-10.md) - 3 篇论文 (V2)
- ...
```

## 🎨 特色功能

### 1. 折叠式设计

- 摘要默认折叠，保持页面简洁
- 点击展开查看详细内容
- 适合快速浏览和深度阅读

### 2. 双语支持

- 英文原文保留完整性
- 中文翻译便于理解
- 专业术语保持英文

### 3. 一键访问

- 所有链接可直接点击
- PDF 直达下载页面
- 项目和代码仓库一目了然

### 4. 版本灵活切换

- 通过 GitHub Variables 控制
- 无需修改代码
- 实时生效

## 📊 性能指标

### V1 基础版

- **运行时间**：2-5 分钟
- **API 调用**：约 10-30 次（取决于论文数量）
- **成本**：极低（Gemini Flash 免费额度充足）
- **适用场景**：日常更新、快速浏览

### V2 增强版

- **运行时间**：10-30 分钟
- **API 调用**：约 50-150 次
- **成本**：中等（仍在免费额度内）
- **适用场景**：周末深度研究、重要论文分析

## 🛡️ 安全性

- **私有仓库**：默认创建私有仓库，保护数据隐私
- **Secrets 加密**：API Key 加密存储在 GitHub Secrets
- **最小权限**：GitHub Actions 仅有必要的读写权限
- **无外部依赖**：不依赖第三方服务器

## 🔄 维护建议

### 日常维护

- **每周检查**：查看 Actions 运行状态
- **关键词优化**：根据实际需求调整筛选关键词
- **版本切换**：平时用 V1，周末用 V2

### 定期更新

- **Python 依赖**：定期更新 `openai` 包
- **系统工具**：GitHub Actions 环境自动更新
- **脚本优化**：根据 arXiv API 变化调整代码

### 成本控制

- **API 配额**：Gemini Flash 有免费额度，通常足够
- **存储空间**：GitHub 免费版有 1GB 存储，足够存储多年数据
- **Actions 时长**：免费版每月 2000 分钟，每天运行 5 分钟，可用 400 天

## 📚 文档索引

| 文档 | 用途 |
|------|------|
| `README.md` | 项目主页，功能介绍 |
| `SETUP.md` | 配置指南，首次部署 |
| `USAGE.md` | 使用指南，日常操作 |
| `PROJECT_SUMMARY.md` | 项目总结，技术细节 |
| `deploy.sh` | 部署脚本，一键推送 |

## 🎯 未来扩展

### 可能的改进方向

1. **多语言支持**：添加更多语言翻译
2. **邮件通知**：新论文发布时发送邮件
3. **RSS 订阅**：生成 RSS feed
4. **统计分析**：论文趋势分析和可视化
5. **自定义筛选**：支持用户自定义筛选规则
6. **论文评分**：AI 自动评估论文质量
7. **相关论文推荐**：基于内容的推荐系统

### 扩展建议

- 保持核心功能简洁
- 新功能作为可选模块
- 避免过度依赖外部服务
- 保持低成本运行

## 💡 最佳实践

1. **首次使用**：先用 V1 测试，确认正常后再考虑 V2
2. **关键词调整**：根据实际筛选结果优化关键词列表
3. **定时任务**：选择 arXiv 更新后的时间（北京时间 10:00 较好）
4. **版本切换**：工作日用 V1，周末用 V2 深度分析
5. **定期检查**：每周查看一次 Actions 状态

## 📞 支持与反馈

- **问题报告**：在 GitHub Issues 中提交
- **功能建议**：欢迎提出改进意见
- **贡献代码**：欢迎 Pull Request

## 📄 许可证

MIT License - 可自由使用、修改和分发

---

**项目创建时间**：2026-02-11
**当前版本**：1.0.0
**维护状态**：活跃开发中
