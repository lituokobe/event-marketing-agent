# ai_gateway_service.py
# AI网关服务 - 负责Lua脚本与AI模型服务之间的通信协调
from collections import defaultdict
import requests
import json
import time
from datetime import datetime
import redis
from flask import Flask, request, jsonify
from config.setting import settings
from common.logger import setup_logger
import threading

app = Flask(__name__)
logger = setup_logger('ai_gateway', category='gateway', console_output=True)

# 服务配置
AI_MODEL_SERVICE_URL = settings.AI_MODEL_SERVICE_URL  # AI模型服务地址
GATEWAY_VERSION = "1.0.0"

# Redis连接
redis_pool = redis.ConnectionPool(
    host=settings.REDIS_SERVER,
    password=settings.REDIS_PASSWORD,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=True,
    max_connections=50
)
redis_client = redis.Redis(connection_pool=redis_pool)


class GatewayManager:
    """网关管理器"""

    def __init__(self):
        # 🎯 直接使用可序列化的数据结构
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'active_tasks': []  # 直接使用列表而不是 set
        }
        self.task_model_map = {}
        self.model_tasks = defaultdict(list)  # 使用列表而不是 set

        # 🎯 内部使用 set 用于快速查找（不暴露给 JSON）
        self._active_tasks_set = set()
        self._model_tasks_set = defaultdict(set)

    def record_call(self, success=True):
        """记录呼叫统计"""
        self.stats['total_calls'] += 1
        if success:
            self.stats['successful_calls'] += 1
        else:
            self.stats['failed_calls'] += 1

    def bind_task_to_model(self, task_id, model_id):
        if task_id in self.task_model_map:
            existing_model_id = self.task_model_map[task_id]
            if existing_model_id != model_id:
                logger.warning(f"任务 {task_id} 从模型 {existing_model_id} 切换到 {model_id}")
                self.unbind_task(task_id)

        # 🎯 同时更新内部 set 和外部列表
        self._active_tasks_set.add(task_id)
        if task_id not in self.stats['active_tasks']:
            self.stats['active_tasks'].append(task_id)

        self._model_tasks_set[model_id].add(task_id)
        if task_id not in self.model_tasks[model_id]:
            self.model_tasks[model_id].append(task_id)

        self.task_model_map[task_id] = model_id
        logger.info(f"🔗 任务绑定到模型 - 任务: {task_id}, 模型: {model_id}")

    def unbind_task(self, task_id):
        # 🎯 同时更新内部 set 和外部列表
        if task_id in self._active_tasks_set:
            self._active_tasks_set.remove(task_id)
        if task_id in self.stats['active_tasks']:
            self.stats['active_tasks'].remove(task_id)

        if task_id in self.task_model_map:
            model_id = self.task_model_map[task_id]
            del self.task_model_map[task_id]

            if model_id in self._model_tasks_set and task_id in self._model_tasks_set[model_id]:
                self._model_tasks_set[model_id].remove(task_id)
            if model_id in self.model_tasks and task_id in self.model_tasks[model_id]:
                self.model_tasks[model_id].remove(task_id)

    # 🎯 不再需要特殊的序列化方法
    # 因为所有数据结构已经是可序列化的


# 全局网关管理器
gateway_manager = GatewayManager()


def async_initialize_model(model_id, config_data, expire_time):
    """异步初始化模型"""

    def initialize_task():
        try:
            payload = {
                'model_id': model_id,
                'config': config_data or {},
                'expire_time': expire_time
            }

            logger.info(f"🔄 开始异步初始化模型: {model_id}")
            response = requests.post(
                f"{AI_MODEL_SERVICE_URL}/model/initialize",
                json=payload,
                timeout=60  # 初始化可能较慢
            )

            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    logger.info(f"✅ 异步模型初始化成功: {model_id}")
                else:
                    logger.error(f"❌ 异步模型初始化失败: {model_id}, 错误: {result.get('message')}")
            else:
                logger.error(f"❌ 异步模型初始化HTTP错误: {model_id}, 状态码: {response.status_code}")

        except Exception as e:
            logger.error(f"🚨 异步模型初始化异常: {model_id}, 错误: {str(e)}")

    # 启动异步线程
    thread = threading.Thread(target=initialize_task, daemon=True, name=f"AsyncInit-{model_id}")
    thread.start()
    logger.info(f"🚀 提交异步模型初始化任务: {model_id}")


def call_model_service(model_id, backstop_model, user_input, conversation_history, call_id, task_id):
    """调用AI模型服务生成话术 ，返回(响应, 历史, 实际使用的模型ID)"""
    # 完善响应处理******
    # result['response'] = [
    #   {},{},{}, -- 历史对话
    #   {
    #       "role": "assistant", 角色
    #       "content": "", 回复话术
    #       'intention_tag': '', 回复话术所在流程的意向标签
    #       'dialog_id': '', 话术id
    #       'logic': {
    #           'user_logic_title':{'主线流程【肯定】分支 “肯定”', '大模型理解：“客户表示想要了解装修”'},
    #           'assistant_logic_title':'【主线流程】：主流程二业务介绍、肯定 -> 主线流程跳转下一主线流程',
    #           'detail': [
    #               {'master_id':'主流程id','branch_id':'节点id', 'hit_branch_id':'命中的分支的id', 'infer_type': '推理的类型': '1 意图 2 知识库', 'infer_use_id':'意图/知识库id', 'infer_use_type': '知识库的类型1通用问题 2业务问题 3一般问题'},
    #               {'master_id':'主流程id','branch_id':'节点id', 'hit_branch_id':'命中的分支1肯定2否定3拒接4无应答5默认', 'infer_type': '推理的类型': '1 意图 2 知识库', 'infer_use_id':'意图/知识库id', 'infer_use_type': '知识库的类型1通用问题 2业务问题 3一般问题'}
    #           ]
    #       },
    #   }]
    # 还需要 知道 用户的话术是肯定还是否定  用 logic.detail 里最后一条的 hit_branch_id 的类型
    # 主流程完成次数 出现的所有master_id 算次数
    #
    try:
        payload = {
            'model_id': model_id,
            'backstop_model': backstop_model,
            'user_input': user_input,
            'conversation_history': conversation_history,
            'call_id': call_id,
            'task_id': task_id
        }

        start_time = time.time()
        response = requests.post(
            f"{AI_MODEL_SERVICE_URL}/model/generate",
            json=payload,
            timeout=10
        )
        response_time = (time.time() - start_time) * 1000  # 毫秒

        if response.status_code == 200:
            result = response.json()
            if result['success']:
                logger.info(f"🎯 AI响应成功 - 任务: {task_id}, 呼叫: {call_id}, 耗时: {response_time:.1f}ms")
                gateway_manager.record_call(success=True)
                # 🎯 返回实际使用的模型ID
                actual_model_id = result.get('model_id', model_id)
                return result['response'], result['conversation_history'], actual_model_id
            else:
                logger.error(f"❌ AI服务业务错误: {result.get('message')}")
        elif response.status_code == 404:
            logger.error(f"🔍 模型未找到 - 模型: {model_id}, 任务: {task_id}")
            gateway_manager.record_call(success=False)
        else:
            logger.error(f"❌ AI服务HTTP错误: {response.status_code}")

    except requests.exceptions.RequestException as e:
        logger.error(f"🔌 调用AI模型服务失败: {str(e)}")
        gateway_manager.record_call(success=False)
        # 🎯 增强：记录更详细的错误信息
        error_detail = {
            'main_model': model_id,
            'backstop_model': backstop_model,
            'error': str(e),
            'call_id': call_id,
            'task_id': task_id
        }
        logger.error(f"🚨 所有模型都不可用: {json.dumps(error_detail)}")

    # 默认返回兜底模型
    default_response = "您好，系统正在处理中，请稍候。"
    return default_response, conversation_history, backstop_model


def calculate_tts_duration(text, speed=1.0):
    """计算TTS语音时长"""
    if not text:
        return 0
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_chars = len([c for c in text if c.isalpha()])
    punctuation = len([c for c in text if c in '，。！？；：,.!?;:'])

    base_duration = (chinese_chars / 4 + english_chars / 2) / speed
    pause_duration = punctuation * 0.3
    total_duration = base_duration + pause_duration

    return max(1.0, round(total_duration, 2))


@app.route('/gateway/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        # 检查AI模型服务状态
        model_health = requests.get(f"{AI_MODEL_SERVICE_URL}/health", timeout=5)
        model_status = model_health.json() if model_health.status_code == 200 else {'status': 'unreachable'}

        # 检查Redis连接
        redis_status = 'healthy' if redis_client.ping() else 'unhealthy'

    except Exception as e:
        model_status = {'status': f'unreachable: {str(e)}'}
        redis_status = 'unhealthy'

    return jsonify({
        'status': 'healthy',
        'service': 'ai_gateway',
        'version': GATEWAY_VERSION,
        'timestamp': datetime.now().isoformat(),
        'dependencies': {
            'ai_model_service': model_status,
            'redis': redis_status
        },
        'statistics': gateway_manager.stats
    })


@app.route('/gateway/model/start', methods=['POST'])
def start_model():
    """初始化模型接口 - 异步版本"""
    data = request.json
    model_id = data.get('model_id')
    config_data = data.get('config_data', {})
    expire_time = data.get('expire_time')
    only_delay = data.get('only_delay', False)

    if not model_id:
        return jsonify({
            'success': False,
            'message': 'model_id 参数不能为空'
        }), 400

    logger.info(f"🚀 接收模型启动请求 - 模型: {model_id}, 仅延期: {only_delay}")

    if only_delay:
        # 只延期模式 - 同步处理（快速）
        try:
            payload = {
                'model_id': model_id,
                'expire_time': expire_time,
                'action': 'extend_only'
            }
            response = requests.post(
                f"{AI_MODEL_SERVICE_URL}/model/extend",
                json=payload,
                timeout=10
            )
            if response.status_code == 200 and response.json().get('success'):
                return jsonify({
                    'success': True,
                    'message': f'模型 {model_id} 过期时间已延长',
                    'model_id': model_id
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f'模型 {model_id} 延期失败'
                }), 500
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'模型延期请求失败: {str(e)}'
            }), 500
    else:
        # 完整初始化模式 - 异步处理
        async_initialize_model(model_id, config_data, expire_time)
        return jsonify({
            'success': True,
            'message': f'模型 {model_id} 初始化请求已提交，正在后台处理',
            'model_id': model_id,
            'async': True
        })


# 其他接口保持不变...
@app.route('/gateway/conversation', methods=['POST'])
def conversation():
    """对话接口"""
    data = request.json
    call_id = data.get('call_id')
    model_id = data.get('model_id', 'default')
    backstop_model = data.get('backstop_model', 'default')
    task_id = data.get('task_id')
    current_input = data.get('current_input', '')

    if not task_id:
        return jsonify({
            'success': False,
            'message': 'task_id 参数不能为空'
        }), 400

    logger.info(f"📞 处理对话请求 - 任务: {task_id}, 呼叫: {call_id}")

    # 从Redis获取对话历史
    conversation_key = f"call:conversation:{call_id}"
    try:
        existing_conversation = redis_client.get(conversation_key)
    except redis.RedisError as e:
        logger.error(f"🔴 Redis连接异常: {str(e)}")
        # 🎯 降级处理：使用空的历史记录继续处理
        conversation_history = []
        existing_conversation = None

    if existing_conversation:
        conversation_data = json.loads(existing_conversation)
        conversation_history = conversation_data.get('messages', [])
        # 🎯 检查之前是否已经切换到兜底模型
        actual_model_id = conversation_data.get('actual_model_id', model_id)
    else:
        conversation_history = []
        actual_model_id = model_id  # 初始使用主模型
        conversation_data = {
            'call_id': call_id,
            'task_id': task_id,
            'model_id': model_id,
            'actual_model_id': actual_model_id,  # 🎯 新增：记录实际使用的模型
            'backstop_model': backstop_model,
            'start_time': time.time(),
            'messages': conversation_history
        }

    # 调用AI模型服务生成话术
    ai_response, updated_history, used_model_id = call_model_service(
        actual_model_id, backstop_model, current_input, conversation_history, call_id, task_id
    )
    # 🎯 更新实际使用的模型ID（如果发生了切换）
    if used_model_id != actual_model_id:
        old_model_id = actual_model_id
        actual_model_id = used_model_id
        logger.info(f"🔄 模型切换: {old_model_id} -> {actual_model_id}")
    # 更新对话历史到Redis
    conversation_data['messages'] = updated_history
    conversation_data['actual_model_id'] = actual_model_id  # 🎯 更新实际模型
    conversation_data['last_update'] = time.time()
    redis_client.setex(conversation_key, 3600, json.dumps(conversation_data))

    # 自动绑定任务到实际使用的模型
    gateway_manager.bind_task_to_model(task_id, actual_model_id)

    # 构建混合播放内容
    mixed_content = {
        'playback_type': 'tts_only',
        'content': [
            {
                'type': 'tts',
                'value': ai_response,
                'duration': calculate_tts_duration(ai_response)
            }
        ],
        'total_duration': calculate_tts_duration(ai_response),
        'allow_bargein': True
    }

    # 动态ASR参数
    dynamic_params = {
        'asr_no_input_timeout': int((calculate_tts_duration(ai_response) + 2) * 1000),
        'asr_speech_timeout': 15000,
        'asr_silence_threshold': 25,
        'asr_sensitivity': 0.8,
        'tts_voice': 'xiaoyan',
        'tts_speed': 1.0,
        'barge_in_enabled': True,
        'estimated_total_duration': calculate_tts_duration(ai_response)
    }

    response = {
        'success': True,
        'action': 'speak',
        'content': mixed_content,
        'dynamic_params': dynamic_params,
        'next_step': 'wait_input',
        'variables': {},
        'end_call': False,
        'current_turn': len([msg for msg in updated_history if msg['role'] == 'assistant']),
        'task_id': task_id,
        'model_id': actual_model_id,  # 🎯 返回实际使用的模型ID
        'call_id': call_id
    }

    logger.info(f"✅ 对话响应生成 - 任务: {task_id}, 呼叫: {call_id}")
    return jsonify(response)


def check_model_service_health():
    """检查AI模型服务健康状态"""
    try:
        response = requests.get(f"{AI_MODEL_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ AI模型服务状态: {health_data.get('status', 'unknown')}")
            logger.info(f"📊 当前模型数: {health_data.get('model_stats', {}).get('total_models', 0)}")
            return True
        else:
            logger.warning(f"⚠️ AI模型服务响应异常: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ 无法连接到AI模型服务: {str(e)}")
        return False

def start_gateway_service(port=5001):
    """启动AI网关服务"""
    logger.info(f"🚀 启动AI网关服务，端口: {port}")
    logger.info(f"📋 服务版本: {GATEWAY_VERSION}")
    logger.info(f"🔗 AI模型服务: {AI_MODEL_SERVICE_URL}")
    logger.info("✅ 网关服务初始化完成，等待请求...")
    # 🎯 检查依赖服务状态
    if not check_model_service_health():
        logger.warning("⚠️ AI模型服务可能不可用，但网关服务将继续启动")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


if __name__ == '__main__':
    start_gateway_service()