# ai_service.py
from flask import Flask, request, jsonify
import requests
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime, timedelta
import json
import uuid
from collections import defaultdict
import psutil
import os

# from agent_builders.chatflow_builder import build_chatflow
# from config.config_setup import ChatFlowConfig
from functionals.log_utils import logger_chatflow
# from functionals.state import ChatState
from config.setting import settings

app = Flask(__name__)

PHP_CALLBACK_URL = settings.PHP_CALLBACK_URL  # PHP回调地址


# 在 DynamicModelManager 类中添加异步通知机制
class AsyncNotificationManager:
    """异步通知管理器"""

    def __init__(self, max_workers=5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.Queue()
        self.is_running = True
        self._start_worker()

    def _start_worker(self):
        """启动工作线程"""

        def worker():
            while self.is_running:
                try:
                    # 从队列获取任务，超时1秒
                    task = self.task_queue.get(timeout=1)
                    if task is None:  # 停止信号
                        break

                    url, payload, task_type = task
                    self._send_notification(url, payload, task_type)
                    self.task_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    logger_chatflow.error(f"异步通知工作线程异常: {str(e)}")

        # 启动多个工作线程
        for i in range(3):
            thread = threading.Thread(target=worker, daemon=True, name=f"NotifyWorker-{i}")
            thread.start()

    def _send_notification(self, url, payload, task_type):
        """发送通知"""
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=10)
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                logger_chatflow.info(f"📤 异步通知成功 - 类型: {task_type}, 耗时: {response_time:.1f}ms")
            else:
                logger_chatflow.error(f"❌ 异步通知失败 - 类型: {task_type}, 状态码: {response.status_code}")

        except requests.exceptions.RequestException as e:
            logger_chatflow.error(f"🔌 异步通知请求失败 - 类型: {task_type}, 错误: {str(e)}")
        except Exception as e:
            logger_chatflow.error(f"🚨 异步通知异常 - 类型: {task_type}, 错误: {str(e)}")

    def add_notification(self, url, payload, task_type):
        """添加通知任务到队列"""
        try:
            self.task_queue.put((url, payload, task_type), timeout=0.1)
            logger_chatflow.debug(f"📝 添加异步通知任务 - 类型: {task_type}")
        except queue.Full:
            logger_chatflow.warning(f"⚠️ 通知队列已满，丢弃任务 - 类型: {task_type}")

    def shutdown(self):
        """关闭通知管理器"""
        self.is_running = False
        self.executor.shutdown(wait=False)


class DynamicModelManager:
    def __init__(self):
        self.models = {}  # {model_id: model_data}
        self.model_usage = defaultdict(int)  # 模型使用计数
        self.model_last_used = {}  # 模型最后使用时间
        self.model_tasks = defaultdict(set)  # 模型关联的任务
        self.lock = threading.RLock()

        # 异步通知管理器
        self.notification_manager = AsyncNotificationManager(max_workers=5)

        # 资源配置
        self.max_models = 50  # 最大模型数量
        self.model_timeout = 3600  # 模型空闲超时时间(秒)
        self.cleanup_interval = 300  # 清理间隔

        # 启动后台清理线程
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """启动后台清理线程"""

        def cleanup_worker():
            while True:
                time.sleep(self.cleanup_interval)
                self.cleanup_idle_models()

        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()

    def _notify_php_model_activated(self, model_id):
        """异步通知PHP模型激活"""
        try:
            payload = {
                'model_id': model_id,
                'status': 'activated',
                'timestamp': datetime.now().isoformat()
            }
            # 使用异步线程
            thread = threading.Thread(
                target=lambda: requests.post(f"{PHP_CALLBACK_URL}", json=payload, timeout=3),
                daemon=True
            )
            thread.start()
            logger_chatflow.info(f"📤 异步通知PHP模型激活: {model_id}")
        except Exception as e:
            logger_chatflow.error(f"❌ 异步通知PHP模型激活失败: {str(e)}")

    def _notify_php_model_activation_failed(self, model_id, error_msg):
        """通知PHP模型激活失败"""
        try:
            payload = {
                'model_id': model_id,
                'status': 'sleep',  # 回退到休眠状态
                'timestamp': datetime.now().isoformat(),
                'reason': f'activation_failed: {error_msg}'
            }
            # 使用异步线程发送通知
            thread = threading.Thread(
                target=lambda: requests.post(f"{PHP_CALLBACK_URL}", json=payload, timeout=3),
                daemon=True
            )
            thread.start()
            logger_chatflow.info(f"📤 通知PHP模型激活失败: {model_id}, 原因: {error_msg}")
        except Exception as e:
            logger_chatflow.error(f"❌ 通知PHP模型激活失败失败: {str(e)}")

    def _notify_php_model_sleep(self, model_id):
        """异步通知PHP模型休眠"""
        payload = {
            'model_id': model_id,
            'status': 'sleep',
            'timestamp': datetime.now().isoformat(),
            'reason': 'no_active_tasks_or_expired'
        }
        self.notification_manager.add_notification(
            f"{PHP_CALLBACK_URL}",
            payload,
            "model_sleep"
        )

    def notify_php_task_pause(self, task_id, model_id, reason):
        """异步通知PHP暂停任务"""
        try:
            payload = {
                'task_id': task_id,
                'model_id': model_id,
                'status': 'pause_task',
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            # 使用异步线程
            thread = threading.Thread(
                target=lambda: requests.post(f"{PHP_CALLBACK_URL}", json=payload, timeout=3),
                daemon=True
            )
            thread.start()
            logger_chatflow.warning(f"📤 异步通知PHP暂停任务: {task_id}, 原因: {reason}")
        except Exception as e:
            logger_chatflow.error(f"❌ 异步通知PHP暂停任务失败: {str(e)}")

    def initialize_model(self, model_id, config_data=None, task_id=None, expire_time=None):
        """动态初始化模型"""
        with self.lock:
            # 检查是否已达模型上限
            if len(self.models) >= self.max_models:
                # 尝试清理空闲模型
                self.cleanup_idle_models(force=True)
                if len(self.models) >= self.max_models:
                    error_msg = f"模型数量已达上限 {self.max_models}，无法创建新模型"
                    self._notify_php_model_activation_failed(model_id, error_msg)
                    raise Exception(error_msg)

            # 如果模型已存在，增加使用计数
            if model_id in self.models:
                self.model_usage[model_id] += 1
                if task_id:
                    self.model_tasks[model_id].add(task_id)
                if expire_time:
                    self.models[model_id]['expire_time'] = expire_time
                logger_chatflow.info(f"模型 {model_id} 已存在，增加使用计数: {self.model_usage[model_id]}")
                return True

            try:
                logger_chatflow.info(f"开始动态初始化模型: {model_id}")

                # 构建配置
                # chatflow_config = self._build_chatflow_config(config_data)
                # chatflow = build_chatflow(chatflow_config)
                chatflow = {}

                # 存储模型实例
                self.models[model_id] = {
                    'instance': chatflow,
                    'config': config_data or {},
                    'created_time': datetime.now(),
                    'expire_time': expire_time or (time.time() + 14 * 24 * 3600),
                    'memory_usage': self._get_memory_usage(),
                    'status': 'active'
                }

                self.model_usage[model_id] = 1
                self.model_last_used[model_id] = datetime.now()

                if task_id:
                    self.model_tasks[model_id].add(task_id)

                logger_chatflow.info(f"模型 {model_id} 动态初始化成功，当前模型总数: {len(self.models)}")

                # 通知PHP模型激活
                self._notify_php_model_activated(model_id)
                return True

            except Exception as e:
                logger_chatflow.error(f"模型 {model_id} 动态初始化失败: {str(e)}")
                # 清理可能的部分初始化
                if model_id in self.models:
                    del self.models[model_id]
                if model_id in self.model_usage:
                    del self.model_usage[model_id]
                # 通知PHP激活失败
                self._notify_php_model_activation_failed(model_id, str(e))
                raise

    def get_model(self, model_id, task_id=None):
        """获取模型实例，更新使用时间"""
        with self.lock:
            if model_id in self.models:
                # 检查模型是否过期
                if self._check_model_expired(model_id):
                    logger_chatflow.warning(f"模型 {model_id} 已过期")
                    # 通知PHP模型休眠
                    self._notify_php_model_sleep(model_id)
                    return None

                self.model_last_used[model_id] = datetime.now()
                self.model_usage[model_id] += 1
                if task_id and task_id not in self.model_tasks[model_id]:
                    self.model_tasks[model_id].add(task_id)
                return self.models[model_id]['instance']
            return None

    def _check_model_expired(self, model_id):
        """检查模型是否过期"""
        if model_id in self.models:
            model_data = self.models[model_id]
            current_time = time.time()
            if current_time > model_data['expire_time']:
                return True
        return False

    def extend_model_expire_time(self, model_id, expire_time):
        """延长模型过期时间"""
        with self.lock:
            if model_id in self.models:
                self.models[model_id]['expire_time'] = expire_time
                logger_chatflow.info(f"模型 {model_id} 过期时间已延长至: {expire_time}")
                return True
            return False

    def release_model(self, model_id, task_id=None):
        """释放模型使用计数"""
        with self.lock:
            if model_id in self.model_usage:
                self.model_usage[model_id] = max(0, self.model_usage[model_id] - 1)

                # 如果指定了task_id，从任务列表中移除
                if task_id and task_id in self.model_tasks[model_id]:
                    self.model_tasks[model_id].remove(task_id)

                logger_chatflow.info(f"释放模型 {model_id} 使用计数，当前: {self.model_usage[model_id]}")

    def destroy_model(self, model_id, force=False):
        """销毁模型实例"""
        with self.lock:
            if model_id not in self.models:
                return True

            # 检查是否还有任务在使用
            if not force and self.model_usage[model_id] > 0:
                logger_chatflow.warning(f"模型 {model_id} 仍有 {self.model_usage[model_id]} 个任务在使用，无法销毁")
                return False

            try:
                # 清理模型资源
                model_data = self.models[model_id]

                # 从管理器中移除
                del self.models[model_id]
                if model_id in self.model_usage:
                    del self.model_usage[model_id]
                if model_id in self.model_last_used:
                    del self.model_last_used[model_id]
                if model_id in self.model_tasks:
                    del self.model_tasks[model_id]

                logger_chatflow.info(f"模型 {model_id} 已销毁，剩余模型数: {len(self.models)}")

                # 通知PHP模型休眠
                self._notify_php_model_sleep(model_id)
                return True

            except Exception as e:
                logger_chatflow.error(f"销毁模型 {model_id} 失败: {str(e)}")
                return False

    def cleanup_idle_models(self, force=False):
        """清理空闲模型"""
        with self.lock:
            current_time = datetime.now()
            models_to_remove = []

            for model_id, last_used in self.model_last_used.items():
                # 检查是否超时且没有使用
                idle_time = (current_time - last_used).total_seconds()
                if (idle_time > self.model_timeout and self.model_usage[model_id] == 0) or force:
                    models_to_remove.append(model_id)

            for model_id in models_to_remove:
                logger_chatflow.info(f"清理空闲模型: {model_id}, 空闲时间: {idle_time}秒")
                self.destroy_model(model_id, force=True)

            if models_to_remove:
                logger_chatflow.info(f"清理完成，移除了 {len(models_to_remove)} 个空闲模型")

    # def _build_chatflow_config(self, config_data):
    #     """根据配置数据构建chatflow配置"""
    #     # 构建chatflow实例 范本 根据这个样子实例化
    #     # chatflow_data = {
    #     #     "agent_config": config_data.agent_config,
    #     #     "keyword_json": config_data.keyword_json,
    #     #     "semantic_json": config_data.semantic_json,
    #     #     "llm_json": config_data.llm_json,
    #     #     "db_path": DB_PATH,
    #     #     "design_json": config_data.design_json,
    #     # }
    #     # chatflow = build_chatflow(chatflow_data)
    #
    #     # 这里根据实际的config_data构建配置
    #     if config_data and 'agent_config' in config_data:
    #         # 使用自定义配置
    #         chatflow_config = ChatFlowConfig.from_custom_data(
    #             agent_config=config_data.get('agent_config'),
    #             keyword_json=config_data.get('keyword_json'),
    #             semantic_json=config_data.get('semantic_json'),
    #             llm_json=config_data.get('llm_json'),
    #             design_json=config_data.get('design_json')
    #         )
    #     else:
    #         # 使用默认路径配置
    #         chatflow_config = ChatFlowConfig.from_paths(
    #             AGENT_DATA_PATH,
    #             KEYWORD_JSON_PATH,
    #             SEMANTIC_JSON_PATH,
    #             LLM_JSON_PATH,
    #             DB_PATH,
    #             CHATFLOW_DESIGN_PATH
    #         )
    #
    #     return chatflow_config

    def get_model_status(self, model_id=None):
        """获取模型状态"""
        with self.lock:
            if model_id:
                if model_id in self.models:
                    model_data = self.models[model_id]
                    return {
                        'model_id': model_id,
                        'status': 'active',
                        'created_time': model_data['created_time'].isoformat(),
                        'last_used': self.model_last_used[model_id].isoformat(),
                        'usage_count': self.model_usage[model_id],
                        'associated_tasks': self.model_tasks[model_id],
                        'memory_usage': model_data['memory_usage']
                    }
                else:
                    return {'model_id': model_id, 'status': 'not_found'}
            else:
                return {
                    'total_models': len(self.models),
                    'active_models': len([m for m in self.models if self.model_usage[m] > 0]),
                    'models': {
                        model_id: {
                            'created_time': data['created_time'].isoformat(),
                            'last_used': self.model_last_used[model_id].isoformat(),
                            'usage_count': self.model_usage[model_id],
                            'associated_tasks': self.model_tasks[model_id],
                            'status': 'active'
                        }
                        for model_id, data in self.models.items()
                    }
                }

    def _get_memory_usage(self):
        """获取内存使用情况"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB


# 全局动态模型管理器
model_manager = DynamicModelManager()


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    status = model_manager.get_model_status()
    return jsonify({
        'status': 'healthy',
        'service': 'dynamic_ai_service',
        'timestamp': datetime.now().isoformat(),
        'model_stats': {
            'total_models': status['total_models'],
            'active_models': status['active_models']
        },
        'memory_usage': model_manager._get_memory_usage()
    })


@app.route('/model/initialize', methods=['POST'])
def initialize_model():
    """初始化模型接口 - 动态创建"""
    data = request.json
    model_id = data.get('model_id')
    config_data = data.get('config', {})
    task_id = data.get('task_id', None)
    expire_time = data.get('expire_time')  # 需要添加这行
    if not model_id:
        return jsonify({
            'success': False,
            'message': 'model_id 参数不能为空'
        }), 400

    try:
        if model_manager.initialize_model(model_id, config_data, task_id, expire_time):
            return jsonify({
                'success': True,
                'message': f'模型 {model_id} 初始化成功',
                'model_id': model_id,
                'task_id': task_id
            })
        else:
            return jsonify({
                'success': False,
                'message': f'模型 {model_id} 初始化失败'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'模型 {model_id} 初始化异常: {str(e)}'
        }), 500


@app.route('/model/extend', methods=['POST'])
def extend_model():
    """延长模型过期时间接口"""
    data = request.json
    model_id = data.get('model_id')
    expire_time = data.get('expire_time')

    if not model_id or not expire_time:
        return jsonify({
            'success': False,
            'message': 'model_id 和 expire_time 参数不能为空'
        }), 400

    if model_manager.extend_model_expire_time(model_id, expire_time):
        return jsonify({
            'success': True,
            'message': f'模型 {model_id} 过期时间延长成功',
            'model_id': model_id
        })
    else:
        return jsonify({
            'success': False,
            'message': f'模型 {model_id} 未找到，延长失败'
        }), 404


@app.route('/model/generate', methods=['POST'])
def generate_response():
    """生成话术接口"""
    data = request.json
    model_id = data.get('model_id')
    user_input = data.get('user_input', '')
    conversation_history = data.get('conversation_history', [])
    call_id = data.get('call_id', 'unknown')
    task_id = data.get('task_id')

    if not model_id:
        return jsonify({
            'success': False,
            'message': 'model_id 参数不能为空'
        }), 400

    # 获取模型实例
    chatflow = model_manager.get_model(model_id, task_id)
    if not chatflow:
        # 模型未找到或已过期，通知PHP暂停任务
        if task_id:
            model_manager.notify_php_task_pause(task_id, model_id, "model_not_found_or_expired")

        return jsonify({
            'success': False,
            'message': f'模型 {model_id} 未找到或已过期',
            'error_code': 'MODEL_NOT_FOUND'
        }), 404

    try:
        # 构建对话状态
        # state = ChatState(
        #     messages=conversation_history.copy(),
        #     dialog_state=[]
        # )
        state = {
            "messages": [],
            "dialog_state": []
        }

        # 添加用户输入
        if user_input and user_input.strip():
            state["messages"].append({"role": "user", "content": user_input})

        # 配置
        conv_config = {"configurable": {"thread_id": f"call_{call_id}"}}
        state = chatflow.invoke(state, config=conv_config)
        # 生成回复 结构 [历史+最新回复 {"role": "user", "content": "", 'logic': '', 'intention_tag': ''}]
        # state["messages"] 里除了 content 和role 还需要有 命中逻辑 logic 和 当前的意图标签 intention_tag
        # 命中的分支；命中的知识库类型、id、名称 或者 命中的意图id、名称；
        # state["message"] = [
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

        # 提取AI回复
        ai_response = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "assistant":
                ai_response = msg["content"]
                break

        logger_chatflow.info(f"生成话术成功 - 模型: {model_id}, 呼叫: {call_id}, 回复长度: {len(ai_response)}")

        return jsonify({
            'success': True,
            # 当前回复的话术内容
            'response': ai_response,
            # 历史记录
            'conversation_history': state["messages"],
            'call_id': call_id,
            'model_id': model_id,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger_chatflow.error(f"生成话术失败 - 模型: {model_id}, 呼叫: {call_id}, 错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'话术生成失败: {str(e)}'
        }), 500
    finally:
        # 释放模型使用计数
        if task_id:
            model_manager.release_model(model_id, task_id)


@app.route('/model/destroy', methods=['POST'])
def destroy_model():
    """销毁模型接口"""
    data = request.json
    model_id = data.get('model_id')
    task_id = data.get('task_id')
    force = data.get('force', False)

    if not model_id:
        return jsonify({
            'success': False,
            'message': 'model_id 参数不能为空'
        }), 400

    if model_manager.destroy_model(model_id, force):
        return jsonify({
            'success': True,
            'message': f'模型 {model_id} 销毁成功'
        })
    else:
        return jsonify({
            'success': False,
            'message': f'模型 {model_id} 销毁失败，可能仍有任务在使用'
        }), 400


@app.route('/model/status', methods=['GET'])
def get_model_status():
    """获取模型状态"""
    model_id = request.args.get('model_id')
    status = model_manager.get_model_status(model_id)
    return jsonify(status)


@app.route('/model/cleanup', methods=['POST'])
def cleanup_models():
    """手动触发清理空闲模型"""
    data = request.json
    force = data.get('force', False)

    model_manager.cleanup_idle_models(force)

    status = model_manager.get_model_status()
    return jsonify({
        'success': True,
        'message': '空闲模型清理完成',
        'current_stats': {
            'total_models': status['total_models'],
            'active_models': status['active_models']
        }
    })


def start_dynamic_service(port=5002):
    """启动动态模型服务"""
    logger_chatflow.info(f"启动动态AI模型服务，端口: {port}")
    logger_chatflow.info("服务特点: 动态模型管理，按需创建，自动清理")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


if __name__ == '__main__':
    start_dynamic_service()
