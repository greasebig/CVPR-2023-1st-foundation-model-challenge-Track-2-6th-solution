# 快速开始指南

## 🚀 5 分钟部署

### 步骤 1：创建 GitHub 仓库

访问 https://github.com/new 并创建仓库：
- **Repository name**: `arxiv-cv-papers`
- **Description**: `每日自动更新 arXiv 计算机视觉相关论文`
- **Privacy**: Private（推荐）
- 点击 **Create repository**

### 步骤 2：推送代码

```bash
cd /home/ubuntu/arxiv-cv-papers

# 运行部署脚本
./deploy.sh

# 按提示输入你的 GitHub 用户名
```

### 步骤 3：配置 API Key

1. 进入仓库页面
2. **Settings** → **Secrets and variables** → **Actions** → **Secrets**
3. 点击 **New repository secret**
4. 添加：
   - Name: `OPENAI_API_KEY`
   - Value: 你的 OpenAI API Key

### 步骤 4：运行测试

1. 进入 **Actions** 标签页
2. 点击 **I understand my workflows, go ahead and enable them**（如果出现）
3. 选择 **Daily arXiv Papers Update**
4. 点击 **Run workflow**
5. 等待 2-5 分钟

### 步骤 5：查看结果

- 返回仓库主页
- 查看 `papers/` 目录
- 查看 `README.md` 的论文索引

## ✅ 完成！

现在系统会每天北京时间 10:00 自动运行。

## 🔧 可选配置

### 启用 V2 版本（AI 深度分析）

1. **Settings** → **Secrets and variables** → **Actions** → **Variables**
2. 点击 **New repository variable**
3. 添加：
   - Name: `VERSION`
   - Value: `v2`

### 修改运行时间

编辑 `.github/workflows/daily-update.yml`：

```yaml
schedule:
  - cron: '0 2 * * *'  # UTC 02:00 = 北京时间 10:00
```

## 📚 更多文档

- [SETUP.md](SETUP.md) - 详细配置指南
- [USAGE.md](USAGE.md) - 完整使用手册
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 技术细节

## ❓ 常见问题

**Q: 如何获取 OpenAI API Key？**
A: 访问 https://platform.openai.com/api-keys 创建

**Q: 为什么没有新论文？**
A: 可能最近 3 天没有符合条件的新论文，或已被去重

**Q: V1 和 V2 有什么区别？**
A: V1 快速简洁，V2 包含 AI 全文分析和批判性评价

**Q: 如何暂停自动运行？**
A: Actions → Daily arXiv Papers Update → ... → Disable workflow

## 💡 提示

- 首次使用建议选择 V1 版本
- 定期查看 Actions 运行状态
- 根据需求调整关键词和类别

---

**需要帮助？** 查看 [USAGE.md](USAGE.md) 或提交 GitHub Issue
