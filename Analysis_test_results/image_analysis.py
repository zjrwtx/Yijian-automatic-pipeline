# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

from camel.agents import ChatAgent
from camel.messages.base import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import ImageAnalysisToolkit
from camel.types import ModelPlatformType, ModelType

def analyze_images(image_paths):
    """
    分析给定的图片路径列表，返回AI分析结果
    
    Args:
        image_paths (list): 图片路径列表
        
    Returns:
        str: AI分析结果
    """
    model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )

    image_analysis_toolkit = ImageAnalysisToolkit(model=model)

    agent = ChatAgent(
        system_message="You are a helpful assistant.",
        model=model,
        tools=[*image_analysis_toolkit.get_tools()],
    )

    # 构建图片路径字符串
    image_paths_str = "\n".join([f"{i+1}、{path}" for i, path in enumerate(image_paths)])
    
    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content=f'''
            把以下图片的内容进行美观格式的输出：
            {image_paths_str}
            ''',
    )
    response = agent.step(user_msg)
    return response.msgs[0].content

if __name__ == "__main__":
    # 测试代码
    test_images = ["i01.png", "i02.png", "i03.png"]
    result = analyze_images(test_images)
    print(result)
