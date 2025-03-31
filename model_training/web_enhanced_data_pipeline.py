import json
import os
from typing import List, Dict, Any, Optional
import argparse
from tqdm import tqdm

from camel.agents import ChatAgent
from camel.configs import DeepSeekConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.loaders import Firecrawl

class WebEnhancedDataPipeline:
    def __init__(self, temperature: float = 0.2, api_key: Optional[str] = None):
        """
        初始化网页增强的合成数据生成pipeline
        
        Args:
            temperature: 模型温度参数
            api_key: DeepSeek API密钥，如果为None则从环境变量获取
        """
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
        
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            model_config_dict=DeepSeekConfig(temperature=temperature).as_dict(),
        )
        
        self.agent = ChatAgent(
            system_message="你是一个专门用于生成高质量合成数据的助手。你需要根据给定的主题和参考内容生成问题和答案对。数学公式必须使用LaTeX格式。确保数据的生成格式保持一致：JSON格式，以问题为键，答案为值。",
            model=self.model
        )
        
        self.firecrawl = Firecrawl()
    
    def decompose_topic(self, main_topic: str, num_subtopics: int = 5) -> List[str]:
        """
        将主题分解为子主题
        
        Args:
            main_topic: 主要主题
            num_subtopics: 子主题数量
            
        Returns:
            子主题列表
        """
        prompt = f"""
        请将以下主题分解为{num_subtopics}个更具体的子主题：
        
        主题：{main_topic}
        
        请直接返回子主题列表，每行一个子主题，不要有任何额外的解释或编号。
        """
        
        response = self.agent.step(prompt)
        subtopics = [
            line.strip() for line in response.msgs[0].content.strip().split("\n")
            if line.strip()
        ]
        
        # 确保我们有足够的子主题
        if len(subtopics) < num_subtopics:
            remaining = num_subtopics - len(subtopics)
            more_prompt = f"""
            请再提供{remaining}个关于"{main_topic}"的子主题，使其与之前提供的不同。
            直接返回子主题列表，每行一个，不要有任何额外的解释或编号。
            """
            more_response = self.agent.step(more_prompt)
            more_subtopics = [
                line.strip() for line in more_response.msgs[0].content.strip().split("\n")
                if line.strip()
            ]
            subtopics.extend(more_subtopics[:remaining])
        
        return subtopics[:num_subtopics]
    
    def crawl_url(self, url: str) -> str:
        """
        爬取URL内容
        
        Args:
            url: 要爬取的URL
            
        Returns:
            提取的内容
        """
        try:
            response = self.firecrawl.crawl(url=url)
            if response["status"] == "completed" and response["data"]:
                return response["data"][0]["markdown"]
            return ""
        except Exception as e:
            print(f"爬取URL {url} 时出错: {e}")
            return ""
    
    def generate_data_from_content(self, subtopic: str, content: str, num_examples: int) -> Dict[str, str]:
        """
        基于爬取的内容生成数据
        
        Args:
            subtopic: 子主题
            content: 爬取的内容
            num_examples: 要生成的示例数量
            
        Returns:
            问题-答案对的字典
        """
        # 限制内容长度，避免超出模型上下文窗口
        max_content_length = 3000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        prompt = f"""
        请基于以下内容，为主题"{subtopic}"生成{num_examples}个高质量的问题和答案对：
        
        参考内容：
        {content}
        
        请按照以下JSON格式返回结果，不要有任何额外的解释：
        
        {{
            "问题1": "答案1",
            "问题2": "答案2",
            ...
        }}
        
        确保问题具有挑战性和多样性，答案简洁明了。
        
        如果是数学问题，必须使用LaTeX格式来表示数学公式，例如：
        {{
            "What is the coefficient of $x^2y^6$ in the expansion of $\\\\left(\\\\frac{{3}}{{5}}x-\\\\frac{{y}}{{2}}\\\\right)^8$?": "\\\\frac{{63}}{{400}}",
            "how many a in banana?": "3"
        }}
        
        问题和答案应该直接或间接地基于给定的参考内容。
        请保持一致的格式和输出风格。
        """
        
        response = self.agent.step(prompt)
        content = response.msgs[0].content
        
        # 提取JSON部分
        try:
            # 查找第一个{和最后一个}之间的内容
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx+1]
                data = json.loads(json_str)
                return data
            else:
                # 尝试直接解析
                return json.loads(content)
        except json.JSONDecodeError:
            # 如果解析失败，返回空字典
            print(f"无法解析子主题 '{subtopic}' 的JSON响应")
            return {}
    
    def generate_fallback_data(self, subtopic: str, num_examples: int) -> Dict[str, str]:
        """
        在没有URL内容时生成备用数据
        
        Args:
            subtopic: 子主题
            num_examples: 要生成的示例数量
            
        Returns:
            问题-答案对的字典
        """
        prompt = f"""
        请为以下子主题生成{num_examples}个高质量的问题和答案对：
        
        子主题：{subtopic}
        
        请按照以下JSON格式返回结果，不要有任何额外的解释：
        
        {{
            "问题1": "答案1",
            "问题2": "答案2",
            ...
        }}
        
        确保问题具有挑战性和多样性，答案简洁明了。
        
        如果是数学问题，必须使用LaTeX格式来表示数学公式，例如：
        {{
            "What is the coefficient of $x^2y^6$ in the expansion of $\\\\left(\\\\frac{{3}}{{5}}x-\\\\frac{{y}}{{2}}\\\\right)^8$?": "\\\\frac{{63}}{{400}}",
            "how many a in banana?": "3"
        }}
        
        请保持一致的格式和输出风格。
        """
        
        response = self.agent.step(prompt)
        content = response.msgs[0].content
        
        # 提取JSON部分
        try:
            # 查找第一个{和最后一个}之间的内容
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx+1]
                data = json.loads(json_str)
                return data
            else:
                # 尝试直接解析
                return json.loads(content)
        except json.JSONDecodeError:
            # 如果解析失败，返回空字典
            print(f"无法解析子主题 '{subtopic}' 的JSON响应")
            return {}
    
    def generate_synthetic_data(
        self, 
        main_topic: str, 
        total_examples: int, 
        urls: List[str] = None,
        num_subtopics: int = 5,
        use_web_content: bool = True
    ) -> Dict[str, str]:
        """
        生成合成数据的主方法
        
        Args:
            main_topic: 主要主题
            total_examples: 要生成的总示例数量
            urls: 用户提供的URL列表
            num_subtopics: 子主题数量
            use_web_content: 是否使用网页内容增强数据生成
            
        Returns:
            合并后的问题-答案对字典
        """
        # 分解主题为子主题
        subtopics = self.decompose_topic(main_topic, num_subtopics)
        
        # 计算每个子主题需要生成的示例数
        examples_per_subtopic = total_examples // len(subtopics)
        remainder = total_examples % len(subtopics)
        
        # 为每个子主题生成数据
        all_data = {}
        for i, subtopic in enumerate(tqdm(subtopics, desc="生成子主题数据")):
            # 最后一个子主题处理余数
            num_examples = examples_per_subtopic + (remainder if i == len(subtopics) - 1 else 0)
            
            subtopic_data = {}
            
            # 如果启用了网页内容增强且提供了URL
            if use_web_content and urls:
                # 爬取URL内容
                for url in urls:
                    content = self.crawl_url(url)
                    if content:
                        # 基于内容生成数据
                        examples_this_url = num_examples // len(urls)
                        url_data = self.generate_data_from_content(subtopic, content, examples_this_url)
                        subtopic_data.update(url_data)
            
            # 如果没有获得足够的示例，使用备用方法生成
            if len(subtopic_data) < num_examples:
                remaining = num_examples - len(subtopic_data)
                fallback_data = self.generate_fallback_data(subtopic, remaining)
                subtopic_data.update(fallback_data)
            
            # 更新总数据集
            all_data.update(subtopic_data)
            
            # 如果仍然没有获得足够的示例，继续生成
            while len(subtopic_data) < num_examples:
                remaining = num_examples - len(subtopic_data)
                more_prompt = f"""
                请为子主题"{subtopic}"再生成{remaining}个不同的问题和答案对，确保它们与之前提供的不同。
                
                必须按以下JSON格式返回：
                {{
                    "问题1": "答案1",
                    "问题2": "答案2"
                }}
                
                如果是数学问题，必须使用LaTeX格式，例如：
                {{
                    "What is the coefficient of $x^2y^6$ in the expansion of $\\\\left(\\\\frac{{3}}{{5}}x-\\\\frac{{y}}{{2}}\\\\right)^8$?": "\\\\frac{{63}}{{400}}"
                }}
                """
                more_response = self.agent.step(more_prompt)
                try:
                    more_content = more_response.msgs[0].content
                    start_idx = more_content.find('{')
                    end_idx = more_content.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1:
                        json_str = more_content[start_idx:end_idx+1]
                        more_data = json.loads(json_str)
                        subtopic_data.update(more_data)
                        all_data.update(more_data)
                    else:
                        more_data = json.loads(more_content)
                        subtopic_data.update(more_data)
                        all_data.update(more_data)
                except:
                    # 如果解析失败，继续尝试
                    pass
        
        return all_data
    
    def save_to_file(self, data: Dict[str, str], output_file: str):
        """
        将生成的数据保存到文件
        
        Args:
            data: 问题-答案对字典
            output_file: 输出文件路径
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"数据已保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description="网页增强合成数据生成器")
    parser.add_argument("--topic", type=str, required=True, help="主题")
    parser.add_argument("--num_examples", type=int, default=10, help="要生成的示例数量")
    parser.add_argument("--num_subtopics", type=int, default=5, help="子主题数量")
    parser.add_argument("--temperature", type=float, default=0.2, help="模型温度参数")
    parser.add_argument("--output", type=str, default="web_enhanced_data.json", help="输出文件路径")
    parser.add_argument("--api_key", type=str, help="DeepSeek API密钥")
    parser.add_argument("--no_web", action="store_true", help="禁用网页内容增强")
    parser.add_argument("--urls", type=str, nargs='+', help="要爬取的URL列表")
    
    args = parser.parse_args()
    
    pipeline = WebEnhancedDataPipeline(temperature=args.temperature, api_key=args.api_key)
    data = pipeline.generate_synthetic_data(
        main_topic=args.topic,
        total_examples=args.num_examples,
        num_subtopics=args.num_subtopics,
        urls=args.urls,
        use_web_content=not args.no_web
    )
    
    pipeline.save_to_file(data, args.output)

if __name__ == "__main__":
    main() 