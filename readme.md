我的毕业论文项目:基于camel框架等AI技术，用于医学检验的全流程检验项目推荐、检验报告生成、检验灵感发现、数据合成与reasoningmodel训练的多智能体流程pipeline 
## base on CAMEL-AI https://github.com/camel-ai/camel
## 本项目使用了 camel框架，并做了代码修改等，该库遵循 Apache License 2.0。原始代码的版权归属于 camel-ai，许可证副本可在以下链接获取：[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)。



# pipeline流程

1. 输入数据
• 输入方式：
  • 直接传入生成的医患对话
  • 传入临床语音转文本的对话记录
  • 传入患者症状、病历等资料
2. 生成医患对话
• Roleplaying 模式：
  • 基于患者背景资料，生成医患对话
  • 输出：模拟的医患对话内容
3. 推荐项目生成
• Workforce 模式：
  • 每个 Agent 调用 MCP 或 Function Call 工具
  • 实现以下功能：
    ◦ 联网搜索
    ◦ 查询科室指导手册
    ◦ 查询图书
    ◦ 查询类似病人数据库
  • 输出：推荐项目列表
4. 检验结果分析
• 输入：
  • 获取检验结果的图像
• 处理：
  • OCR 转成文本
  • 利用 Roleplay Critic 模式分析
• 输出：
  • 检验报告
7.数据结果结果数据合成拿来训练reasoning small model