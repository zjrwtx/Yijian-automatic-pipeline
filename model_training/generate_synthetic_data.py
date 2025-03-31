#!/usr/bin/env python
import argparse
import os
import json
from typing import Dict, Any, List

from synthetic_data_pipeline import SyntheticDataPipeline
from web_enhanced_data_pipeline import WebEnhancedDataPipeline

def display_banner():
    """显示程序横幅"""
    banner = """
    ====================================================
    自动合成数据生成pipeline
    基于大语言模型的高质量数据自动合成工具
    ====================================================
    """
    print(banner)

def write_metadata(
    output_file: str, 
    args: Dict[str, Any], 
    total_examples: int, 
    subtopics_info: Dict[str, int],
    used_urls: List[str] = None
) -> None:
    """
    写入元数据文件，记录生成数据的相关信息
    
    Args:
        output_file: 输出文件路径
        args: 命令行参数
        total_examples: 生成的总示例数
        subtopics_info: 每个子主题生成的示例数统计
        used_urls: 使用的URL列表
    """
    metadata = {
        "main_topic": args.topic,
        "total_examples": total_examples,
        "num_subtopics": args.num_subtopics,
        "pipeline_type": "web_enhanced" if args.web_enabled else "basic",
        "temperature": args.temperature,
        "subtopics_statistics": subtopics_info,
        "output_file": args.output
    }
    
    if used_urls:
        metadata["used_urls"] = used_urls
    
    metadata_file = f"{os.path.splitext(output_file)[0]}_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"元数据已保存到 {metadata_file}")

def load_urls_from_file(file_path: str) -> List[str]:
    """
    从文件加载URL列表
    
    Args:
        file_path: URL文件路径
        
    Returns:
        URL列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip() and line.strip().startswith('http')]
    return urls

def main():
    parser = argparse.ArgumentParser(description="自动合成数据生成工具")
    parser.add_argument("--topic", type=str, required=True, help="要生成数据的主题")
    parser.add_argument("--num_examples", type=int, default=10, help="要生成的总示例数量")
    parser.add_argument("--num_subtopics", type=int, default=5, help="将主题分解为几个子主题")
    parser.add_argument("--temperature", type=float, default=0.2, help="模型温度参数")
    parser.add_argument("--output", type=str, default="synthetic_data.json", help="输出文件路径")
    parser.add_argument("--api_key", type=str, help="DeepSeek API密钥")
    parser.add_argument("--web_enabled", action="store_true", help="启用网页内容增强")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    parser.add_argument("--urls", type=str, nargs='+', help="要爬取的URL列表")
    parser.add_argument("--url_file", type=str, help="包含URL列表的文件路径")
    
    args = parser.parse_args()
    
    display_banner()
    
    print(f"主题: {args.topic}")
    print(f"示例数量: {args.num_examples}")
    print(f"子主题数量: {args.num_subtopics}")
    
    # 处理URL
    urls = []
    if args.url_file:
        try:
            urls = load_urls_from_file(args.url_file)
            print(f"从文件 {args.url_file} 加载了 {len(urls)} 个URL")
        except Exception as e:
            print(f"加载URL文件时出错: {e}")
    
    if args.urls:
        urls.extend(args.urls)
        print(f"命令行提供了 {len(args.urls)} 个URL")
    
    if urls:
        args.web_enabled = True
    
    print(f"使用pipeline类型: {'网页增强型' if args.web_enabled else '基础型'}")
    if args.web_enabled and urls:
        print(f"将使用 {len(urls)} 个URL进行内容增强")
    print("开始生成数据...")
    
    # 根据参数选择合适的pipeline
    if args.web_enabled:
        print("使用网页增强型pipeline...")
        pipeline = WebEnhancedDataPipeline(temperature=args.temperature, api_key=args.api_key)
        data = pipeline.generate_synthetic_data(
            main_topic=args.topic,
            total_examples=args.num_examples,
            num_subtopics=args.num_subtopics,
            urls=urls,
            use_web_content=True
        )
    else:
        print("使用基础型pipeline...")
        pipeline = SyntheticDataPipeline(temperature=args.temperature, api_key=args.api_key)
        data = pipeline.generate_synthetic_data(
            main_topic=args.topic,
            total_examples=args.num_examples,
            num_subtopics=args.num_subtopics
        )
    
    # 保存数据
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"成功生成 {len(data)} 个数据样本")
    print(f"数据已保存到: {args.output}")
    
    # 示例统计子主题数据分布（实际项目中需要从pipeline中获取真实子主题信息）
    # 这里仅做示例，实际使用时需要修改
    subtopics_info = {f"子主题_{i+1}": args.num_examples // args.num_subtopics for i in range(args.num_subtopics)}
    
    # 写入元数据
    write_metadata(args.output, args, len(data), subtopics_info, urls if urls else None)
    
    print("数据生成完成!")

if __name__ == "__main__":
    main() 