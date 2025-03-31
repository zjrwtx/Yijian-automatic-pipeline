#!/usr/bin/env python
"""
示例脚本：演示如何使用合成数据生成pipeline
"""

import os
import json
from synthetic_data_pipeline import SyntheticDataPipeline
from web_enhanced_data_pipeline import WebEnhancedDataPipeline

def run_basic_example():
    """运行基础示例，不使用URL"""
    print("\n=== 基础示例 ===\n")
    
    # 设置API密钥
    # 推荐使用环境变量，也可以直接传递给SyntheticDataPipeline构造函数
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("警告: 未设置DEEPSEEK_API_KEY环境变量")
        print("可以通过以下命令设置环境变量:")
        print("export DEEPSEEK_API_KEY='your_api_key_here'")
        return False

    # 创建pipeline实例
    pipeline = SyntheticDataPipeline(temperature=0.2, api_key=api_key)
    
    # 定义生成参数
    main_topic = "基础数学问题"
    total_examples = 3
    num_subtopics = 2
    
    print(f"开始为主题 '{main_topic}' 生成 {total_examples} 个示例...")
    
    # 生成合成数据
    data = pipeline.generate_synthetic_data(
        main_topic=main_topic,
        total_examples=total_examples,
        num_subtopics=num_subtopics
    )
    
    # 打印生成的数据
    print("\n生成的数据示例:")
    for i, (question, answer) in enumerate(data.items(), 1):
        print(f"\n示例 {i}:")
        print(f"问题: {question}")
        print(f"答案: {answer}")
        if i >= 3:  # 只打印前3个示例
            print("\n...")
            break
    
    # 保存数据到文件
    output_file = "example_synthetic_data.json"
    pipeline.save_to_file(data, output_file)
    
    print(f"\n成功生成 {len(data)} 个数据样本")
    print(f"数据已保存到: {output_file}")
    
    return True

def run_url_example():
    """运行URL增强示例"""
    print("\n=== URL增强示例 ===\n")
    
    # 设置API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("警告: 未设置DEEPSEEK_API_KEY环境变量")
        return False

    # 创建pipeline实例
    pipeline = WebEnhancedDataPipeline(temperature=0.2, api_key=api_key)
    
    # 定义生成参数
    main_topic = "人工智能基础"
    total_examples = 3
    num_subtopics = 1
    
    # 定义URL列表
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
    ]
    
    print(f"开始为主题 '{main_topic}' 使用URL增强生成 {total_examples} 个示例...")
    print(f"使用URL: {urls[0]}")
    
    # 生成合成数据
    data = pipeline.generate_synthetic_data(
        main_topic=main_topic,
        total_examples=total_examples,
        num_subtopics=num_subtopics,
        urls=urls
    )
    
    # 打印生成的数据
    print("\n生成的数据示例:")
    for i, (question, answer) in enumerate(data.items(), 1):
        print(f"\n示例 {i}:")
        print(f"问题: {question}")
        print(f"答案: {answer}")
        if i >= 3:  # 只打印前3个示例
            print("\n...")
            break
    
    # 保存数据到文件
    output_file = "example_url_enhanced_data.json"
    pipeline.save_to_file(data, output_file)
    
    print(f"\n成功生成 {len(data)} 个URL增强型数据样本")
    print(f"数据已保存到: {output_file}")
    
    return True

def main():
    # 运行基础示例
    basic_success = run_basic_example()
    
    # 运行URL增强示例
    if basic_success:
        url_success = run_url_example()
    
    # 使用提示
    print("\n=== 使用完整功能 ===\n")
    print("使用基础pipeline:")
    print("python generate_synthetic_data.py --topic \"您的主题\" --num_examples 10 --num_subtopics 5")
    
    print("\n使用URL增强pipeline:")
    print("python generate_synthetic_data.py --topic \"您的主题\" --num_examples 10 --urls \"https://example.com/page1\" \"https://example.com/page2\"")
    
    print("\n从文件加载URL:")
    print("python generate_synthetic_data.py --topic \"您的主题\" --num_examples 10 --url_file urls.txt")

if __name__ == "__main__":
    main()
