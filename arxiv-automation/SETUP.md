# 项目配置指南

## 📦 部署步骤

### 1. 创建 GitHub 仓库

在本地执行以下命令（如果 token 权限不足，请在 GitHub 网页端手动创建）：

```bash
# 方式 1: 使用 GitHub CLI（推荐）
gh repo create arxiv-cv-papers --private --description "每日自动更新 arXiv 计算机视觉相关论文（Video Generation & Editing）"

# 方式 2: 在 GitHub 网页端手动创建
# 访问 https://github.com/new
# Repository name: arxiv-cv-papers
# Description: 每日自动更新 arXiv 计算机视觉相关论文（Video Generation & Editing）
# Private: ✓
```

### 2. 推送代码到 GitHub

```bash
cd /home/ubuntu/arxiv-cv-papers

# 初始化 Git 仓库
git init
git add .
git commit -m "Initial commit: arXiv CV papers automation system"

# 添加远程仓库（替换为你的用户名）
git remote add origin https://github.com/YOUR_USERNAME/arxiv-cv-papers.git

# 推送代码
git branch -M main
git push -u origin main
```

### 3. 配置 GitHub Secrets

进入仓库的 **Settings → Secrets and variables → Actions → Secrets**，添加以下 Secret：

| Secret 名称 | 值 | 说明 |
|------------|-----|------|
| `OPENAI_API_KEY` | 你的 OpenAI API Key | 用于调用 Gemini 2.5 Flash 模型进行翻译和分析 |

### 4. 配置版本选择（可选）

进入仓库的 **Settings → Secrets and variables → Actions → Variables**，添加以下 Variable：

| Variable 名称 | 值 | 说明 |
|--------------|-----|------|
| `VERSION` | `v1` 或 `v2` | 默认为 `v1`（基础版），设置为 `v2` 启用 AI 深度分析 |

**版本说明**：
- **V1**（基础版）：快速运行，提供论文标题、摘要、链接和中文翻译
- **V2**（增强版）：下载 PDF 全文，AI 深度分析论文质量、实验可靠性和实用价值

### 5. 测试运行

1. 进入仓库的 **Actions** 标签页
2. 选择 `Daily arXiv Papers Update` 工作流
3. 点击 **Run workflow** 按钮
4. 选择 `main` 分支
5. 点击绿色的 **Run workflow** 按钮

等待几分钟，查看运行结果。

### 6. 验证自动化

- 检查 `papers/` 目录是否生成了新的 Markdown 文件
- 检查 `README.md` 是否更新了论文索引
- 确认定时任务已启用（每天北京时间 10:00 自动运行）

## 🔧 自定义配置

### 修改运行时间

编辑 `.github/workflows/daily-update.yml` 文件：

```yaml
on:
  schedule:
    # 修改这里的 cron 表达式
    # 格式: '分 时 日 月 周' (UTC 时间)
    # 例如：北京时间 10:00 = UTC 02:00
    - cron: '0 2 * * *'
```

### 修改检查天数

编辑 `scripts/fetch_papers_v1.py` 或 `scripts/fetch_papers_v2.py`：

```python
DAYS_TO_CHECK = 3      # 检查最近 N 天的论文
DAYS_TO_COMPARE = 5    # 与最近 N 天对比去重
```

### 修改筛选关键词

编辑脚本中的 `VIDEO_KEYWORDS` 列表：

```python
VIDEO_KEYWORDS = [
    "video generation",
    "video editing",
    # 添加更多关键词...
]
```

### 修改 arXiv 类别

编辑脚本中的 `CATEGORIES` 列表：

```python
CATEGORIES = ["cs.CV", "cs.AI", "cs.MM", "cs.RO", "cs.LG"]
```

## 🐛 故障排查

### 问题 1: GitHub Actions 运行失败

**检查项**：
1. 确认 `OPENAI_API_KEY` Secret 已正确配置
2. 查看 Actions 日志中的错误信息
3. 检查 API 配额是否用尽

### 问题 2: 没有新论文更新

**可能原因**：
1. 最近 3 天内确实没有符合条件的新论文
2. 所有论文都已在最近 5 天内收录过（去重机制）
3. 关键词筛选过于严格

**解决方案**：
- 手动运行脚本查看详细日志
- 调整 `DAYS_TO_CHECK` 和 `VIDEO_KEYWORDS`

### 问题 3: V2 版本运行时间过长

**原因**：V2 需要下载 PDF 并进行 AI 分析，每篇论文约需 10-30 秒

**解决方案**：
- 使用 V1 版本（更快）
- 减少 `DAYS_TO_CHECK` 的天数
- 优化关键词筛选，减少匹配的论文数量

## 📞 技术支持

如有问题，请在 GitHub Issues 中提出。

## 📄 许可证

MIT License
