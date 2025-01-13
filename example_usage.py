from text_segmentation import TextSegmenter
import csv
import os

def segment_file_to_csv(input_file_path, output_csv_path, max_length=1000, overlap_length=100, min_segment_length=200):
    # Create text segmenter instance
    segmenter = TextSegmenter(
        delimiter="\n\n",  # 使用段落分隔符
        max_length=max_length,  # 每段最大长度
        overlap_length=overlap_length,  # 重叠长度
        replace_continuous_spaces=True,  # 替换连续空格
        remove_urls=False,  # 保留URL
        min_segment_length=min_segment_length  # 最小段落长度
    )
    
    # Read text from file
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    # Segment the text
    segments = segmenter.segment_text(text)
    
    # Write segments to CSV
    try:
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['段落序号', '文本内容', '字符长度', '包含重叠'])
            
            # Write segments
            for i, segment in enumerate(segments, 1):
                has_overlap = i > 1 and len(segments) > 1
                writer.writerow([i, segment, len(segment), "是" if has_overlap else "否"])
        
        print(f"\n分段结果已保存到: {output_csv_path}")
        print(f"总共分成 {len(segments)} 段")
        print(f"平均段落长度: {sum(len(s) for s in segments) / len(segments):.2f} 字符")
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")
    
    return segments

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = "sample_text.txt"
    output_file = "segmented_text.csv"
    
    # 执行分段并保存为CSV
    segments = segment_file_to_csv(
        input_file_path=input_file,
        output_csv_path=output_file,
        max_length=1000,  # 每段最大1000字符
        overlap_length=100,  # 100字符的重叠
        min_segment_length=200  # 最小段落长度200字符
    )
