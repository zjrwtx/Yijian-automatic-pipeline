我的毕业论文项目:基于camel框架等AI技术，用于医学检验的全流程检验项目推荐、检验报告生成、检验灵感发现、数据合成与reasoningmodel训练的多智能体流程pipeline 
## base on CAMEL-AI https://github.com/camel-ai/camel
## 本项目使用了 camel框架，并做了代码修改等，该库遵循 Apache License 2.0。原始代码的版权归属于 camel-ai，许可证副本可在以下链接获取：[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)。



# pipeline流程


输入数据
• 输入方式：
  • 直接传入生成的医患语音对话或文本对话 
  • 传入患者症状、病历等资料
生成医患对话（合成数据）
• Roleplaying 模式：
  • 基于患者背景资料，生成医患对话
  • 输出：模拟的医患对话内容
推荐项目生成
• Workforce 模式：
  • 每个 Agent 调用  Function Call 工具
  • 实现以下功能：
 ◦ 联网搜索（查询相似病例）
 ◦ 查询医院有的检验项目（数据库）
  • 输出：推荐项目列表


检验结果分析
• 输入：
  • 获取检验结果的图像
• 处理：
  • GPT4o图像识别 转成文本
  • 利用 Roleplay模式分析
• 输出：
  • 检验报告
科研灵感分析
• 基于历史对话数据：
  • 进行科研灵感分析
• 验证灵感：
  • 调用 PubMed MCP 等工具
  • 验证是否已有类似研究
  • 根据参考文献推测新的科研灵感
• 输出：
  • 科研灵感列表
  • 新的科研灵感推测
 6.检验数据自动分析：
基于模型代码生成与执行、对检验科csv等数据进行分析得出科研结论
输出结果
• 最终输出：
  • 医患对话内容
  • 推荐项目列表
  • 检验报告
  • 科研灵感分析结果

7.数据结果通过数据合成管道拿来训练reasoning small model


![image](https://github.com/user-attachments/assets/dc6def2d-6421-4f57-a65b-089a849f7fe3)
