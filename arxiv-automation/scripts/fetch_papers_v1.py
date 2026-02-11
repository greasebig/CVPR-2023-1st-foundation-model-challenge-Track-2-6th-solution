#!/usr/bin/env python3
"""
arXiv Papers Fetcher V1
每日抓取 arXiv 计算机视觉相关论文，筛选 video generation/edit 相关内容
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from openai import OpenAI

# 配置
CATEGORIES = ["cs.CV", "cs.AI", "cs.MM", "cs.RO", "cs.LG"]
DAYS_TO_CHECK = 3  # 检查最近 3 天的论文
DAYS_TO_COMPARE = 5  # 与最近 5 天对比去重
VIDEO_KEYWORDS = [
    "video generation", "video synthesis", "video editing", "video edit",
    "video diffusion", "text-to-video", "image-to-video", "video-to-video",
    "video understanding", "video model", "temporal", "motion generation",
    "video quality", "video enhancement", "video restoration", "video prediction"
]

# 初始化 OpenAI 客户端
# 优先使用 DEEPSEEK_API_KEY，如果没有则回退到默认配置
import os
api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
base_url = "https://api.deepseek.com" if os.environ.get("DEEPSEEK_API_KEY") else None
model_name = "deepseek-chat" if os.environ.get("DEEPSEEK_API_KEY") else "gemini-2.5-flash"

client = OpenAI(api_key=api_key, base_url=base_url)

def fetch_arxiv_papers(category, days=3, max_retries=3):
    """从 arXiv API 获取指定类别的论文"""
    base_url = "http://export.arxiv.org/api/query?"
    
    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # 构建查询
    query = f"cat:{category}"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 300,  # 获取更多结果以确保覆盖
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    url = base_url + urllib.parse.urlencode(params)
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            # arXiv 建议请求间隔至少 3 秒
            if attempt > 0:
                wait_time = 5 * attempt
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            
            # 解析 XML
            root = ET.fromstring(data)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            
            papers = []
            for entry in root.findall("atom:entry", namespace):
                # 提取发布日期
                published = entry.find("atom:published", namespace).text
                pub_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                
                # 只保留指定日期范围内的论文
                if pub_date < start_date:
                    continue
                
                # 提取论文信息
                paper = {
                    "id": entry.find("atom:id", namespace).text.split("/abs/")[-1],
                    "title": entry.find("atom:title", namespace).text.strip().replace("\n", " "),
                    "summary": entry.find("atom:summary", namespace).text.strip().replace("\n", " "),
                    "authors": [author.find("atom:name", namespace).text for author in entry.findall("atom:author", namespace)],
                    "published": pub_date.strftime("%Y-%m-%d"),
                    "pdf_url": entry.find("atom:id", namespace).text.replace("/abs/", "/pdf/"),
                    "abs_url": entry.find("atom:id", namespace).text,
                    "categories": [cat.attrib["term"] for cat in entry.findall("atom:category", namespace)]
                }
                
                papers.append(paper)
            
            return papers
        
        except Exception as e:
            print(f"  尝试 {attempt + 1}/{max_retries} 失败: {e}")
            if attempt == max_retries - 1:
                print(f"Error fetching papers for {category}: 达到最大重试次数")
                return []
            continue
    
    return []

def is_video_related(paper):
    """判断论文是否与视频相关"""
    text = (paper["title"] + " " + paper["summary"]).lower()
    
    for keyword in VIDEO_KEYWORDS:
        if keyword.lower() in text:
            return True
    
    return False

def extract_links(paper):
    """从摘要中提取项目和代码链接"""
    text = paper["summary"]
    
    # 常见的链接模式
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    
    links = {
        "project": None,
        "code": None
    }
    
    for url in urls:
        url_lower = url.lower()
        if "github.com" in url_lower or "gitlab.com" in url_lower:
            if not links["code"]:
                links["code"] = url
        elif "project" in url_lower or "page" in url_lower or "site" in url_lower:
            if not links["project"]:
                links["project"] = url
        elif not links["project"]:
            # 其他链接默认作为项目链接
            links["project"] = url
    
    return links

def translate_text(text, target_lang="zh"):
    """使用 AI 翻译文本"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the following academic text to Chinese. Keep technical terms in English when appropriate. Only return the translation, no explanations."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def load_recent_papers(days=5):
    """加载最近几天的论文 ID 用于去重"""
    # 自动获取当前脚本所在目录的父目录下的 papers 文件夹
    papers_dir = Path(__file__).parent.parent / "papers"
    recent_ids = set()
    
    if not papers_dir.exists():
        return recent_ids
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for md_file in papers_dir.glob("*.md"):
        try:
            file_date = datetime.strptime(md_file.stem, "%Y-%m-%d")
            if file_date >= start_date:
                content = md_file.read_text(encoding="utf-8")
                # 提取 arXiv ID
                ids = re.findall(r'arxiv\.org/abs/(\d+\.\d+)', content)
                recent_ids.update(ids)
        except:
            continue
    
    return recent_ids

def generate_markdown(papers, date_str):
    """生成 Markdown 格式的论文列表"""
    md_content = f"# arXiv Papers - {date_str}\n\n"
    md_content += f"**更新时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"**论文数量**: {len(papers)}\n\n"
    md_content += "---\n\n"
    
    for i, paper in enumerate(papers, 1):
        # 翻译标题和摘要
        print(f"Translating paper {i}/{len(papers)}: {paper['id']}")
        title_zh = translate_text(paper["title"])
        summary_zh = translate_text(paper["summary"])
        
        # 提取链接
        links = extract_links(paper)
        
        md_content += f"## {i}. {paper['title']}\n\n"
        md_content += f"**中文标题**: {title_zh}\n\n"
        md_content += f"**作者**: {', '.join(paper['authors'][:5])}"
        if len(paper['authors']) > 5:
            md_content += f" et al."
        md_content += "\n\n"
        md_content += f"**发布日期**: {paper['published']}\n\n"
        md_content += f"**arXiv ID**: [{paper['id']}]({paper['abs_url']})\n\n"
        md_content += f"**PDF**: [下载链接]({paper['pdf_url']})\n\n"
        
        # 添加项目和代码链接
        if links["project"]:
            md_content += f"**项目主页**: {links['project']}\n\n"
        if links["code"]:
            md_content += f"**代码仓库**: {links['code']}\n\n"
        
        md_content += f"**类别**: {', '.join(paper['categories'])}\n\n"
        
        # 英文摘要（折叠）
        md_content += "<details>\n"
        md_content += "<summary><b>摘要 (Abstract)</b></summary>\n\n"
        md_content += f"{paper['summary']}\n\n"
        md_content += "</details>\n\n"
        
        # 中文摘要（折叠）
        md_content += "<details>\n"
        md_content += "<summary><b>中文摘要</b></summary>\n\n"
        md_content += f"{summary_zh}\n\n"
        md_content += "</details>\n\n"
        
        md_content += "---\n\n"
        
        # 避免 API 限流
        time.sleep(1)
    
    return md_content

def update_readme_index():
    """更新 README 中的论文索引"""
    base_dir = Path(__file__).parent.parent
    papers_dir = base_dir / "papers"
    readme_path = base_dir / "README.md"
    
    if not papers_dir.exists():
        return
    
    # 获取所有论文文件并按日期排序
    paper_files = sorted(papers_dir.glob("*.md"), reverse=True)
    
    index_content = "\n"
    for paper_file in paper_files:
        date_str = paper_file.stem
        # 读取文件获取论文数量
        content = paper_file.read_text(encoding="utf-8")
        count_match = re.search(r'\*\*论文数量\*\*: (\d+)', content)
        count = count_match.group(1) if count_match else "0"
        
        index_content += f"- [{date_str}](papers/{paper_file.name}) - {count} 篇论文\n"
    
    # 更新 README
    readme_content = readme_path.read_text(encoding="utf-8")
    pattern = r'<!-- PAPERS_INDEX_START -->.*?<!-- PAPERS_INDEX_END -->'
    replacement = f'<!-- PAPERS_INDEX_START -->{index_content}<!-- PAPERS_INDEX_END -->'
    updated_content = re.sub(pattern, replacement, readme_content, flags=re.DOTALL)
    
    readme_path.write_text(updated_content, encoding="utf-8")

def main():
    print("=" * 60)
    print("arXiv Papers Fetcher V1")
    print("=" * 60)
    
    # 加载最近的论文 ID 用于去重
    print(f"\n加载最近 {DAYS_TO_COMPARE} 天的论文 ID...")
    recent_ids = load_recent_papers(DAYS_TO_COMPARE)
    print(f"已加载 {len(recent_ids)} 个论文 ID")
    
    # 获取所有类别的论文
    all_papers = []
    for i, category in enumerate(CATEGORIES):
        print(f"\n正在获取 {category} 类别的论文...")
        papers = fetch_arxiv_papers(category, DAYS_TO_CHECK)
        print(f"获取到 {len(papers)} 篇论文")
        all_papers.extend(papers)
        
        # 在请求之间添加延迟（arXiv 建议至少 3 秒）
        if i < len(CATEGORIES) - 1:
            print("  等待 3 秒...")
            time.sleep(3)
    
    # 去重（基于 arXiv ID）
    unique_papers = {}
    for paper in all_papers:
        if paper["id"] not in unique_papers:
            unique_papers[paper["id"]] = paper
    
    print(f"\n去重后共 {len(unique_papers)} 篇论文")
    
    # 筛选视频相关论文
    video_papers = [p for p in unique_papers.values() if is_video_related(p)]
    print(f"筛选出 {len(video_papers)} 篇视频相关论文")
    
    # 再次去重：排除最近已收录的论文
    new_papers = [p for p in video_papers if p["id"] not in recent_ids]
    print(f"排除已收录论文后，剩余 {len(new_papers)} 篇新论文")
    
    if len(new_papers) == 0:
        print("\n没有新论文需要更新")
        return
    
    # 按发布日期排序
    new_papers.sort(key=lambda x: x["published"], reverse=True)
    
    # 生成 Markdown
    date_str = datetime.now().strftime("%Y-%m-%d")
    print(f"\n生成 Markdown 文件: {date_str}.md")
    md_content = generate_markdown(new_papers, date_str)
    
    # 保存文件
    papers_dir = Path(__file__).parent.parent / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    output_file = papers_dir / f"{date_str}.md"
    output_file.write_text(md_content, encoding="utf-8")
    print(f"已保存到: {output_file}")
    
    # 更新 README 索引
    print("\n更新 README 索引...")
    update_readme_index()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
