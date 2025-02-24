import os
import time
import logging
from google import genai
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff

# ========================
# 日志系统初始化
# ========================
logger = logging.getLogger('api_calls')
logger.setLevel(logging.DEBUG)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'api_calls_{time.strftime("%Y%m%d")}.log')
print(f"Creating log file at: {log_file}")

try:
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_handler.setLevel(logging.DEBUG)
    print("Successfully created file handler")
except Exception as e:
    print(f"Error creating file handler: {str(e)}")
    
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.debug("Logger initialization completed")
logger.info("API logging system started")

# ========================
# 常用图标定义
# ========================
SUCCESS_ICON = "✓"
ERROR_ICON = "✗"
WAIT_ICON    = "⟳"

# ========================
# 数据类定义，用于返回结果格式化
# ========================
@dataclass
class ChatMessage:
    content: str

@dataclass
class ChatChoice:
    message: ChatMessage

@dataclass
class ChatCompletion:
    choices: list[ChatChoice]

# ========================
# 加载环境变量
# ========================
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")

# ========================
# 根据配置选择 API 提供方与相应的 API Key、模型
# ========================
api_provider = os.getenv("API_PROVIDER", "deepseek")

if api_provider.lower() == "deepseek":
    deepseek_api_key = os.getenv("DEEPEEK_API_KEY")
    if not deepseek_api_key:
        logger.error(f"{ERROR_ICON} 未找到 DEEPEEK_API_KEY 环境变量")
        raise ValueError("DEEPEEK_API_KEY not found in environment variables")
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    logger.info(f"{SUCCESS_ICON} DEEPSEEK 客户端初始化成功")
    # deepseek 模型通常固定为 deepseek-chat
    model_name = "deepseek-chat"

elif api_provider.lower() == "gemini":
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")
    if not gemini_api_key:
        logger.error(f"{ERROR_ICON} 未找到 GEMINI_API_KEY 环境变量")
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    if not model_name:
        model_name = "gemini-1.5-flash"
        logger.info(f"{WAIT_ICON} 使用默认 Gemini 模型: {model_name}")
    client = genai.Client(api_key=gemini_api_key)
    logger.info(f"{SUCCESS_ICON} Gemini 客户端初始化成功")

else:
    logger.error(f"{ERROR_ICON} 未知的 API_PROVIDER: {api_provider}")
    raise ValueError("Unsupported API_PROVIDER value")

# ========================
# 带重试机制的 API 调用函数
# ========================
@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    max_time=300,
    giveup=lambda e: "AFC is enabled" not in str(e)
)
def generate_content_with_retry(model, contents, config=None):
    """
    带重试机制的内容生成函数
    :param model: 模型名称
    :param contents: 用户输入内容
    :param config: 配置参数，通常包含系统指令
    :return: API 调用响应
    """
    try:
        logger.info(f"{WAIT_ICON} 正在调用 {api_provider.upper()} API...")
        logger.info(f"请求内容: {contents[:500]}..." if len(str(contents)) > 500 else f"请求内容: {contents}")
        logger.info(f"请求配置: {config}")

        if api_provider.lower() == "deepseek":
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": str(config)},
                    {"role": "user",   "content": str(contents)},
                ],
                stream=False
            )
        elif api_provider.lower() == "gemini":
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        else:
            raise ValueError("Unsupported API_PROVIDER")

        logger.info(f"{SUCCESS_ICON} API 调用成功")
        logger.info(f"响应内容: {response.text[:500]}..." if len(str(response.text)) > 500 else f"响应内容: {response.text}")
        return response

    except Exception as e:
        if "AFC is enabled" in str(e):
            logger.warning(f"{ERROR_ICON} 触发 API 限制，等待重试... 错误: {e}")
            time.sleep(5)
            raise e
        logger.error(f"{ERROR_ICON} API 调用失败: {e}")
        logger.error(f"错误详情: {e}")
        raise e

# ========================
# 获取聊天完成结果（包含重试逻辑）
# ========================
def get_chat_completion(messages, model=None, max_retries=3, initial_retry_delay=1):
    """
    获取聊天完成结果，包含重试逻辑
    :param messages: 消息列表，包含 role 与 content
    :param model: 使用的模型名称，如为空则使用默认 model_name
    :param max_retries: 最大重试次数
    :param initial_retry_delay: 初始重试延时（秒）
    :return: 聊天回复内容，或 None 表示失败
    """
    try:
        if model is None:
            model = model_name
        logger.info(f"{WAIT_ICON} 使用模型: {model}")
        logger.debug(f"消息内容: {messages}")

        for attempt in range(max_retries):
            try:
                # 整合 prompt 与系统指令
                prompt = ""
                system_instruction = None
                for message in messages:
                    role = message["role"]
                    content = message["content"]
                    if role == "system":
                        system_instruction = content
                    elif role == "user":
                        prompt += f"User: {content}\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n"
                config = {}
                if system_instruction:
                    config['system_instruction'] = system_instruction

                # 调用带重试机制的 API
                response = generate_content_with_retry(
                    model=model,
                    contents=prompt.strip(),
                    config=config
                )
                if response is None:
                    logger.warning(f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries}: API 返回空值")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    return None

                # 格式化返回结果
                chat_message = ChatMessage(content=response.text)
                chat_choice = ChatChoice(message=chat_message)
                completion = ChatCompletion(choices=[chat_choice])
                logger.debug(f"API 原始响应: {response.text}")
                logger.info(f"{SUCCESS_ICON} 成功获取响应")
                return completion.choices[0].message.content

            except Exception as e:
                logger.error(f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    retry_delay = initial_retry_delay * (2 ** attempt)
                    logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"{ERROR_ICON} 最终错误: {e}")
                    return None

    except Exception as e:
        logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {e}")
        return None
