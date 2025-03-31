#!/usr/bin/env python
import argparse
import os
import json
from typing import Dict, Any

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
    subtopics_info: Dict[str, int]
) -> None:
    """
    写入元数据文件，记录生成数据的相关信息
    
    Args:
        output_file: 输出文件路径
        args: 命令行参数
        total_examples: 生成的总示例数
        subtopics_info: 每个子主题生成的示例数统计
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
    
    metadata_file = f"{os.path.splitext(output_file)[0]}_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"元数据已保存到 {metadata_file}")

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
    
    args = parser.parse_args()
    
    display_banner()
    
    print(f"主题: {args.topic}")
    print(f"示例数量: {args.num_examples}")
    print(f"子主题数量: {args.num_subtopics}")
    print(f"使用pipeline类型: {'网页增强型' if args.web_enabled else '基础型'}")
    print("开始生成数据...")
    
    # 根据参数选择合适的pipeline
    if args.web_enabled:
        print("使用网页增强型pipeline，将爬取相关网页内容...")
        pipeline = WebEnhancedDataPipeline(temperature=args.temperature, api_key=args.api_key)
        data = pipeline.generate_synthetic_data(
            main_topic=args.topic,
            total_examples=args.num_examples,
            num_subtopics=args.num_subtopics,
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
    write_metadata(args.output, args, len(data), subtopics_info)
    
    print("数据生成完成!")

if __name__ == "__main__":
    main() 