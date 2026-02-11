# 使用指南

## 📖 快速开始

### 第一步：创建 GitHub 仓库

由于 GitHub token 权限限制，您需要手动创建仓库：

**方式 1：使用 GitHub 网页**
1. 访问 https://github.com/new
2. Repository name: `arxiv-cv-papers`
3. Description: `每日自动更新 arXiv 计算机视觉相关论文（Video Generation & Editing）`
4. 选择 **Private**（推荐）或 Public
5. 点击 **Create repository**

**方式 2：使用 GitHub CLI**（如果有足够权限）
```bash
gh repo create arxiv-cv-papers --private --description "每日自动更新 arXiv 计算机视觉相关论文"
```

### 第二步：推送代码

```bash
cd /home/ubuntu/arxiv-cv-papers

# 设置默认分支为 main
git branch -M main

# 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/arxiv-cv-papers.git

# 推送代码
git push -u origin main
```

### 第三步：配置 GitHub Secrets

1. 进入仓库页面
2. 点击 **Settings** → **Secrets and variables** → **Actions**
3. 点击 **Secrets** 标签页
4. 点击 **New repository secret**
5. 添加以下 Secret：

| Name | Value | 说明 |
|------|-------|------|
| `OPENAI_API_KEY` | 你的 OpenAI API Key | 用于 Gemini 2.5 Flash 模型 |

**获取 API Key**：
- 如果你已经有 OpenAI API key，可以直接使用
- 或者访问 https://platform.openai.com/api-keys 创建新的 API key

### 第四步：配置版本选择（可选）

1. 在 **Settings** → **Secrets and variables** → **Actions** 页面
2. 点击 **Variables** 标签页
3. 点击 **New repository variable**
4. 添加变量：

| Name | Value | 说明 |
|------|-------|------|
| `VERSION` | `v1` 或 `v2` | 不设置则默认为 `v1` |

**版本对比**：

| 特性 | V1 基础版 | V2 增强版 |
|------|----------|----------|
| 论文标题 | ✅ | ✅ |
| 论文摘要 | ✅ | ✅ |
| 中文翻译 | ✅ | ✅ |
| 项目/代码链接 | ✅ | ✅ |
| PDF 全文下载 | ❌ | ✅ |
| AI 深度分析 | ❌ | ✅ |
| 批判性评价 | ❌ | ✅ |
| 运行时间 | ~2-5 分钟 | ~10-30 分钟 |
| API 调用成本 | 低 | 中等 |

**推荐**：
- 日常使用：**V1**（快速、低成本）
- 深度研究：**V2**（详细分析、批判性评价）

### 第五步：测试运行

1. 进入仓库的 **Actions** 标签页
2. 如果看到提示 "Workflows aren't being run on this repository"，点击 **I understand my workflows, go ahead and enable them**
3. 选择左侧的 **Daily arXiv Papers Update** 工作流
4. 点击右侧的 **Run workflow** 下拉按钮
5. 确认分支为 `main`
6. 点击绿色的 **Run workflow** 按钮

等待几分钟，查看运行结果：
- ✅ 绿色勾：运行成功
- ❌ 红色叉：运行失败（点击查看日志）

### 第六步：查看结果

运行成功后：
1. 返回仓库主页
2. 查看 `papers/` 目录，应该有新的 Markdown 文件（如 `2026-02-11.md`）
3. 查看 `README.md`，论文索引应该已更新

## 🔧 高级配置

### 修改运行时间

编辑 `.github/workflows/daily-update.yml`：

```yaml
on:
  schedule:
    # 格式：'分 时 日 月 周' (UTC 时间)
    # 北京时间 = UTC + 8
    # 例如：北京时间 10:00 = UTC 02:00
    - cron: '0 2 * * *'
```

**常用时间对照表**：

| 北京时间 | UTC 时间 | Cron 表达式 |
|---------|---------|------------|
| 08:00 | 00:00 | `0 0 * * *` |
| 10:00 | 02:00 | `0 2 * * *` |
| 12:00 | 04:00 | `0 4 * * *` |
| 18:00 | 10:00 | `0 10 * * *` |
| 22:00 | 14:00 | `0 14 * * *` |

### 修改检查范围

编辑 `scripts/fetch_papers_v1.py` 和 `scripts/fetch_papers_v2.py`：

```python
# 在文件开头的配置部分

DAYS_TO_CHECK = 3      # 检查最近 N 天的论文
DAYS_TO_COMPARE = 5    # 与最近 N 天对比去重
```

**建议**：
- `DAYS_TO_CHECK`：1-3 天（太多会导致重复）
- `DAYS_TO_COMPARE`：5-7 天（确保去重效果）

### 修改筛选关键词

编辑脚本中的 `VIDEO_KEYWORDS` 列表：

```python
VIDEO_KEYWORDS = [
    "video generation",
    "video synthesis",
    "video editing",
    "video edit",
    "video diffusion",
    "text-to-video",
    "image-to-video",
    # 添加你关心的关键词
    "video captioning",
    "video summarization",
    # ...
]
```

### 修改 arXiv 类别

编辑脚本中的 `CATEGORIES` 列表：

```python
CATEGORIES = [
    "cs.CV",  # Computer Vision
    "cs.AI",  # Artificial Intelligence
    "cs.MM",  # Multimedia
    "cs.RO",  # Robotics
    "cs.LG",  # Machine Learning
    # 可以添加更多类别
]
```

**常用 arXiv 类别**：
- `cs.CV` - Computer Vision and Pattern Recognition
- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning
- `cs.CL` - Computation and Language (NLP)
- `cs.GR` - Graphics
- `cs.MM` - Multimedia
- `cs.RO` - Robotics

## 🐛 故障排查

### 问题 1：Actions 运行失败，提示 "OPENAI_API_KEY not found"

**原因**：未配置 Secret

**解决**：
1. 检查 Settings → Secrets and variables → Actions → Secrets
2. 确认 `OPENAI_API_KEY` 已添加
3. 重新运行 workflow

### 问题 2：运行成功但没有新论文

**可能原因**：
1. 最近 3 天内确实没有符合条件的新论文
2. 所有论文都已在最近 5 天内收录（去重机制生效）
3. 关键词筛选过于严格

**解决方案**：
1. 查看 Actions 日志，确认筛选了多少论文
2. 调整 `DAYS_TO_CHECK` 增加检查范围
3. 扩展 `VIDEO_KEYWORDS` 关键词列表

### 问题 3：arXiv API 返回 429 错误

**原因**：请求过于频繁，触发 arXiv 限流

**解决**：
- 脚本已内置重试机制和延迟（每次请求间隔 3 秒）
- 如果仍然失败，可能是网络问题，等待几分钟后重试
- GitHub Actions 环境通常不会遇到此问题

### 问题 4：V2 版本运行超时

**原因**：V2 需要下载 PDF 并进行 AI 分析，耗时较长

**解决**：
1. 使用 V1 版本（更快）
2. 减少 `DAYS_TO_CHECK` 的天数
3. 优化关键词筛选，减少匹配的论文数量
4. GitHub Actions 免费版有 6 小时超时限制，通常足够

### 问题 5：翻译质量不佳

**原因**：使用的是 Gemini 2.5 Flash 模型，翻译质量可能不够完美

**解决**：
- 可以修改脚本中的翻译 prompt
- 或者使用其他翻译 API（需要修改代码）

## 📊 监控和维护

### 查看运行历史

1. 进入 **Actions** 标签页
2. 查看所有运行记录
3. 点击具体运行查看详细日志

### 手动触发运行

1. 进入 **Actions** 标签页
2. 选择 **Daily arXiv Papers Update**
3. 点击 **Run workflow**

### 暂停自动运行

1. 进入 **Actions** 标签页
2. 选择 **Daily arXiv Papers Update**
3. 点击右上角的 **...** 菜单
4. 选择 **Disable workflow**

### 恢复自动运行

1. 进入 **Actions** 标签页
2. 找到被禁用的 workflow
3. 点击 **Enable workflow**

## 💡 最佳实践

### 1. 定期检查

- 每周查看一次 Actions 运行状态
- 确保没有持续失败的任务

### 2. 合理设置运行时间

- 避开 arXiv 高峰时段（美国东部时间工作时间）
- 推荐：北京时间早上 10:00（UTC 02:00）

### 3. 版本选择策略

- **平时**：使用 V1，快速获取论文列表
- **周末**：切换到 V2，深度分析本周重要论文

### 4. 关键词优化

- 定期审查筛选出的论文
- 根据实际需求调整关键词列表
- 避免关键词过于宽泛导致噪音

### 5. 备份重要内容

- GitHub 自动保存所有历史版本
- 可以随时回溯查看之前的论文

## 📞 获取帮助

如有问题：
1. 查看 Actions 日志中的错误信息
2. 参考本文档的故障排查部分
3. 在 GitHub Issues 中提问

## 📄 许可证

MIT License - 可自由使用、修改和分发
