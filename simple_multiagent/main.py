from openai import OpenAI
import time
import json

client = OpenAI()

class Agent:
    def __init__(self, name, expertise, system_prompt):
        self.name = name
        self.expertise = expertise  # 领域专长（示例：算法/故事/科学）
        self.system_prompt = system_prompt
        self.last_active = 0  # 最后发言时间戳
        self.tasks = []  # 记录分配给该Agent的任务

class DialogueSystem:
    def __init__(self):
        self.message_pipe = []  # 中央消息管道
        self.agents = []
        self.speaker_history = []  # 记录发言顺序
        self.collaboration_topic = ""  # 当前协作主题
        self.subtasks = []  # 拆分的子任务列表
        self.progress = {}  # 各子任务的进度
        
    def add_agent(self, name, expertise, prompt):
        self.agents.append(Agent(name, expertise, prompt))
    
    def _select_speaker(self):
        """智能选择下一个发言的agent"""
        agent_list = "\n".join([f"{agent.name}（专长：{agent.expertise}）" for agent in self.agents])
        subtask_status = "\n".join([f"子任务{i+1}: {task}" for i, task in enumerate(self.subtasks)])
        
        # 使用GPT分析对话上下文
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": f"""根据当前协作主题、子任务和agent专长选择最合适的发言人。

                当前协作主题: {self.collaboration_topic}
                当前子任务列表: 
                {subtask_status}
                
                当前对话：
                {self._format_history()}
                
                可选agent：
                {agent_list}
                
                请按以下规则决策：
                1. 优先选择专长与当前讨论子任务最匹配的agent
                2. 当agent被@提及时，优先选择
                3. 长时间未发言的agent优先考虑
                4. 确保所有agent都有机会参与协作
                5. 直接回复用户提问
                
                只需返回agent名字，不要任何解释。"""
            }]
        )
        selected = response.choices[0].message.content.strip()
        return next((i for i, a in enumerate(self.agents) if a.name == selected), 0)
    
    def _format_history(self, max_length=1500):
        """格式化最近的历史记录，避免超出token限制"""
        history = []
        total_len = 0
        for msg in reversed(self.message_pipe):
            msg_str = f"{msg['speaker']}: {msg['content']}"
            if total_len + len(msg_str) > max_length:
                break
            history.insert(0, msg_str)
            total_len += len(msg_str)
        return "\n".join(history)
    
    def _decompose_topic(self, topic):
        """将主题分解为多个子任务"""
        agent_list = "\n".join([f"{agent.name}（专长：{agent.expertise}）" for agent in self.agents])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": f"""请根据以下主题和可用的专家，将任务分解为3-5个具体的子任务，以便专家们协作完成。
                
                主题: {topic}
                
                可用专家:
                {agent_list}
                
                请确保每个子任务都是明确的，并考虑到专家的专长领域。以JSON格式返回，格式为:
                {{
                    "subtasks": [
                        "子任务1描述",
                        "子任务2描述",
                        ...
                    ]
                }}
                """
            }]
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("subtasks", [])
        except:
            # 如果解析失败，至少返回一个默认任务
            return ["讨论与解决主题相关问题"]
    
    def _assign_tasks(self):
        """根据专长分配子任务给agent"""
        if not self.subtasks:
            return
            
        agent_list = "\n".join([f"{agent.name}（专长：{agent.expertise}）" for agent in self.agents])
        subtask_list = "\n".join([f"子任务{i+1}: {task}" for i, task in enumerate(self.subtasks)])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": f"""请根据专家的专长，为每个子任务分配最合适的主要负责人。
                
                子任务列表:
                {subtask_list}
                
                可用专家:
                {agent_list}
                
                以JSON格式返回，格式为:
                {{
                    "assignments": [
                        {{"task": "子任务1描述", "assigned_to": "专家名称"}},
                        {{"task": "子任务2描述", "assigned_to": "专家名称"}},
                        ...
                    ]
                }}
                """
            }]
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            assignments = result.get("assignments", [])
            
            # 清空所有agent的任务列表
            for agent in self.agents:
                agent.tasks = []
                
            # 分配新任务
            for assignment in assignments:
                task = assignment.get("task")
                assignee = assignment.get("assigned_to")
                
                for agent in self.agents:
                    if agent.name == assignee:
                        agent.tasks.append(task)
                        break
                        
            # 记录任务分配信息到消息管道
            assignment_msg = "任务分配如下:\n"
            for agent in self.agents:
                if agent.tasks:
                    tasks_str = "\n- ".join(agent.tasks)
                    assignment_msg += f"\n【{agent.name}】负责:\n- {tasks_str}"
            
            self.message_pipe.append({
                "speaker": "系统",
                "content": assignment_msg,
                "timestamp": time.time()
            })
            print(f"\n【系统】{assignment_msg}")
            
        except Exception as e:
            print(f"任务分配出错: {e}")
    
    def _check_collaboration_progress(self):
        """检查协作进度，确定是否完成或需要推进"""
        history = self._format_history(max_length=2000)
        subtask_list = "\n".join([f"子任务{i+1}: {task}" for i, task in enumerate(self.subtasks)])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": f"""请分析当前协作进度，确定每个子任务的完成情况。
                
                协作主题: {self.collaboration_topic}
                子任务列表:
                {subtask_list}
                
                对话历史:
                {history}
                
                请以JSON格式返回每个子任务的完成进度(0-100%)和整体协作完成度:
                {{
                    "subtask_progress": [
                        {{"task": "子任务1描述", "progress": 进度百分比}},
                        {{"task": "子任务2描述", "progress": 进度百分比}},
                        ...
                    ],
                    "overall_progress": 整体进度百分比,
                    "next_focus": "下一步应该关注的子任务或方向"
                }}
                """
            }]
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            progress_msg = "当前协作进度:\n"
            
            # 更新进度信息
            self.progress = result
            
            # 子任务进度
            for task_progress in result.get("subtask_progress", []):
                task = task_progress.get("task")
                progress = task_progress.get("progress")
                progress_msg += f"- {task}: {progress}%\n"
            
            # 整体进度
            overall = result.get("overall_progress", 0)
            progress_msg += f"\n整体完成度: {overall}%"
            
            # 下一步建议
            next_focus = result.get("next_focus", "")
            if next_focus:
                progress_msg += f"\n\n建议下一步: {next_focus}"
            
            self.message_pipe.append({
                "speaker": "系统",
                "content": progress_msg,
                "timestamp": time.time()
            })
            print(f"\n【系统】{progress_msg}")
            
            return overall >= 95  # 返回是否完成(95%以上视为完成)
            
        except Exception as e:
            print(f"进度检查出错: {e}")
            return False
    
    def start_collaboration(self, topic, max_turns=30):
        """启动一个新的协作对话"""
        self.collaboration_topic = topic
        self.subtasks = self._decompose_topic(topic)
        
        # 记录主题和子任务到消息管道
        task_msg = f"开始新的协作主题: {topic}\n\n拆分的子任务如下:"
        for i, task in enumerate(self.subtasks):
            task_msg += f"\n{i+1}. {task}"
        
        self.message_pipe.append({
            "speaker": "系统",
            "content": task_msg,
            "timestamp": time.time()
        })
        print(f"\n【系统】{task_msg}")
        
        # 分配任务
        self._assign_tasks()
        
        # 启动对话
        return self.run_dialogue(max_turns=max_turns)
    
    def run_dialogue(self, max_turns=30):
        """运行多轮对话"""
        turn = 0
        while turn < max_turns:
            # 每5轮检查一次协作进度
            if turn > 0 and turn % 5 == 0:
                is_completed = self._check_collaboration_progress()
                if is_completed:
                    print("\n【系统】协作任务已基本完成!")
                    summary = self._generate_summary()
                    self.message_pipe.append({
                        "speaker": "系统",
                        "content": summary,
                        "timestamp": time.time()
                    })
                    print(f"\n【系统】{summary}")
                    break
            
            # 选择发言人
            speaker_idx = self._select_speaker()
            speaker = self.agents[speaker_idx]
            
            # 构造对话上下文
            context = [{"role": "system", "content": 
                f"""{speaker.system_prompt}

当前你正在参与一个协作任务，主题是: {self.collaboration_topic}

子任务列表:
{chr(10).join([f"{i+1}. {task}" for i, task in enumerate(self.subtasks)])}

你被分配的任务:
{chr(10).join([f"- {task}" for task in speaker.tasks])}

请记住，你是一个团队成员，应积极推动对话，提出建设性意见，并与其他专家协作解决问题。当谈论到你的专长领域时，应提供详细专业的见解。你可以直接提问或回应其他专家(@专家名称)。
"""}]
            
            # 添加最近的对话记录作为上下文
            context += [
                {"role": "user" if msg["speaker"] in ["User", "系统"] else "assistant",
                 "content": f"{msg['speaker']}说：{msg['content']}"}
                for msg in self.message_pipe[-8:]  # 增加到最近8条作为上下文
            ]
            
            # 生成回复
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=context
            )
            reply = response.choices[0].message.content
            
            # 记录消息
            self.message_pipe.append({
                "speaker": speaker.name,
                "content": reply,
                "timestamp": time.time()
            })
            speaker.last_active = time.time()
            
            print(f"\n【{speaker.name}】{reply}")
            time.sleep(1)  # 避免速率限制
            
            turn += 1
            
        return self.message_pipe
    
    def _generate_summary(self):
        """生成协作结果摘要"""
        history = self._format_history(max_length=3000)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": f"""请为以下协作讨论生成一份简洁的摘要，总结主要观点、结论和成果。
                
                协作主题: {self.collaboration_topic}
                子任务列表:
                {chr(10).join([f"{i+1}. {task}" for i, task in enumerate(self.subtasks)])}
                
                对话历史:
                {history}
                
                请生成一份全面但简洁的摘要，包括:
                1. 主要讨论点
                2. 每个子任务的关键成果
                3. 总体结论和建议
                """
            }]
        )
        
        return "📋 协作总结:\n\n" + response.choices[0].message.content

# 初始化系统
system = DialogueSystem()

# 用户自定义agent
num_agents = int(input("请输入agent数量（建议3-5个）："))
for i in range(num_agents):
    name = input(f"Agent{i+1}名称：")
    expertise = input(f"{name}的专长领域：")
    prompt = input(f"{name}的角色设定：")
    system.add_agent(name, expertise, prompt)

# 启动协作对话
collaboration_topic = input("请输入协作主题或任务：")
max_turns = int(input("请输入最大对话回合数（建议30-50）："))
system.start_collaboration(collaboration_topic, max_turns=max_turns)