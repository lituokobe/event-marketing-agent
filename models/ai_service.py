# ai_service.py
from flask import Flask, request, jsonify
import requests
import threading
import time
from datetime import datetime, timedelta
import json
from collections import defaultdict
import psutil
import os
import shutil
from agent_builders.chatflow_builder import build_chatflow
from config.config_setup import ChatFlowConfig
from config.paths import AGENT_DATA_PATH, INTENTION_PATH, KNOWLEDGE_PATH, DB_PATH, CHATFLOW_DESIGN_PATH, DIALOG_PATH
from functionals.log_utils import logger_chatflow
from functionals.state import ChatState
from config.setting import settings
from models.async_notification_manager import AsyncNotificationManager
from models.persistence_manager import ModelPersistenceManager

app = Flask(__name__)

PHP_CALLBACK_URL = settings.PHP_CALLBACK_URL  # PHP回调地址


class DynamicModelManager:
    def __init__(self):
        self.models = {}  # {model_id: model_data}
        self.model_usage = defaultdict(int)  # 模型使用计数
        self.model_last_used = {}  # 模型最后使用时间
        self.model_tasks = defaultdict(set)  # 模型关联的任务
        self.model_created_time = {}  # 🎯 记录模型创建时间
        self.lock = threading.RLock()

        # 🎯 持久化管理器
        self.persistence_manager = ModelPersistenceManager()
        # 异步通知管理器
        self.notification_manager = AsyncNotificationManager(max_workers=5)

        # 资源配置
        self.max_models = 50
        self.max_memory_mb = 4096  # 🎯 新增：最大内存限制 2GB
        self.model_timeout = 3600
        self.cleanup_interval = 300
        self.model_memory_estimate = 100  # 🎯 每个模型预估内存占用(MB)

        # 🎯 启动时恢复模型（简化版）
        self._recover_models_on_startup()

        # 启动后台清理线程
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """启动后台清理线程"""

        def cleanup_worker():
            while True:
                time.sleep(self.cleanup_interval)
                try:
                    # 🎯 先检查是否需要紧急清理
                    emergency_result = self.check_and_cleanup_if_needed()
                    if emergency_result.get('cleaned', False):
                        logger_chatflow.debug(f"🔄 紧急清理完成: {emergency_result}")

                    # 🎯 然后进行常规清理
                    normal_result = self.cleanup_idle_models()  # 正常清理
                    if normal_result['removed_count'] > 0:
                        logger_chatflow.debug(f"🔄 常规清理完成: 移除了 {normal_result['removed_count']} 个模型")

                except Exception as e:
                    logger_chatflow.error(f"清理线程异常: {str(e)}")

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

    def _recover_models_on_startup(self):
        """服务启动时恢复模型 - 简化版本"""
        logger_chatflow.info("🔄 开始恢复持久化的模型...")

        # 加载模型配置
        model_configs = self.persistence_manager.load_model_configs()
        recovered_count = 0
        expired_count = 0

        for model_id, config_data in model_configs.items():
            try:
                # 检查模型是否过期
                current_time = time.time()
                expire_time = config_data.get('expire_time', 0)

                if current_time > expire_time:
                    logger_chatflow.info(f"🗑️ 跳过过期模型并删除配置: {model_id}")
                    self.persistence_manager.delete_model_config(model_id)
                    expired_count += 1
                    continue

                # 重新初始化模型
                chatflow_config = self._build_chatflow_config(config_data.get('config', {}))
                chatflow = build_chatflow(chatflow_config)

                # 恢复模型数据
                self.models[model_id] = {
                    'instance': chatflow,
                    'config': config_data.get('config', {}),
                    'created_time': config_data.get('created_time', datetime.now()),
                    'expire_time': expire_time,
                    'memory_usage': config_data.get('memory_usage', 0),
                    'status': 'recovered'
                }

                # 恢复使用统计（重置为0，因为服务重启）
                self.model_usage[model_id] = 0
                self.model_last_used[model_id] = config_data.get('last_used', datetime.now())
                self.model_created_time[model_id] = config_data.get('created_time', datetime.now())

                recovered_count += 1
                logger_chatflow.info(f"✅ 恢复模型成功: {model_id}")

            except Exception as e:
                logger_chatflow.error(f"❌ 恢复模型失败 {model_id}: {str(e)}")
                continue

        logger_chatflow.info(f"🎉 模型恢复完成: 成功 {recovered_count} 个, 过期 {expired_count} 个")

    def initialize_model(self, model_id, config_data=None, task_id=None, expire_time=None):
        """动态初始化模型"""
        with self.lock:
            # 检查是否已达模型上限
            if len(self.models) >= self.max_models:
                # 尝试清理空闲模型
                self.cleanup_idle_models(force_reason='count_exceed')
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
                chatflow_config = self._build_chatflow_config(config_data)
                chatflow = build_chatflow(chatflow_config)

                # 存储模型实例
                current_time = datetime.now()
                self.models[model_id] = {
                    'instance': chatflow,
                    'config': config_data or {},
                    'created_time': current_time,
                    'expire_time': expire_time or (time.time() + 14 * 24 * 3600),
                    'memory_usage': self._get_memory_usage(),
                    'status': 'active'
                }
                self.model_usage[model_id] = 0  # 🎯 修改：新模型初始计数为0
                self.model_last_used[model_id] = current_time
                self.model_created_time[model_id] = current_time

                if task_id:
                    self.model_tasks[model_id].add(task_id)
                    self.model_usage[model_id] += 1  # 🎯 如果有task_id，才增加计数

                logger_chatflow.info(f"模型 {model_id} 动态初始化成功，当前模型总数: {len(self.models)}")

                # 🎯 持久化模型配置
                model_persist_data = {
                    'config': config_data or {},
                    'created_time': current_time,
                    'expire_time': expire_time or (time.time() + 14 * 24 * 3600),
                    'memory_usage': self._get_memory_usage(),
                    'status': 'active',
                    'last_used': current_time
                }
                self.persistence_manager.save_model_config(model_id, model_persist_data)

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
                    # 通知PHP模型休眠    这步和下面的 暂停并重新激活重复 notify_php_task_pause
                    # self._notify_php_model_sleep(model_id)
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

                # 🎯 删除持久化配置（会自动清理旧备份）
                self.persistence_manager.delete_model_config(model_id)

                logger_chatflow.info(f"模型 {model_id} 已销毁，剩余模型数: {len(self.models)}")

                # 通知PHP模型休眠
                self._notify_php_model_sleep(model_id)
                return True

            except Exception as e:
                logger_chatflow.error(f"销毁模型 {model_id} 失败: {str(e)}")
                return False

    def cleanup_idle_models(self, force_reason=None):
        """智能清理空闲模型
        Args:
            force_reason: 强制清理原因 - 'count_exceed', 'memory_exceed', 'manual'
        """
        with self.lock:
            current_time = datetime.now()
            current_memory = self._get_memory_usage()
            total_models = len(self.models)

            # 🎯 收集统计信息
            stats = {
                'total_models': total_models,
                'active_models': len([m for m in self.models if self.model_usage[m] > 0]),
                'idle_models': len([m for m in self.models if self.model_usage[m] == 0]),
                'current_memory_mb': current_memory,
                'max_memory_mb': self.max_memory_mb,
                'max_models': self.max_models,
                'force_reason': force_reason
            }

            logger_chatflow.info(f"📊 清理前状态: 模型总数={stats['total_models']}, "
                                 f"活跃={stats['active_models']}, 空闲={stats['idle_models']}, "
                                 f"内存={stats['current_memory_mb']:.1f}MB")

            # 🎯 按创建时间排序的空闲模型列表（最老的在前）
            idle_models = []
            for model_id in self.models:
                if self.model_usage[model_id] == 0:  # 只考虑空闲模型
                    idle_time = (current_time - self.model_last_used[model_id]).total_seconds()
                    created_time = self.model_created_time[model_id]
                    idle_models.append({
                        'model_id': model_id,
                        'created_time': created_time,
                        'idle_time': idle_time,
                        'last_used': self.model_last_used[model_id]
                    })

            # 按创建时间排序（最老的在前）
            idle_models.sort(key=lambda x: x['created_time'])

            models_to_remove = []
            removal_reason = "normal_timeout"

            # 🎯 智能清理决策
            if force_reason == 'count_exceed':
                # 模型数量超限：清理最老的空闲模型，直到数量达标
                target_count = max(1, self.max_models - 5)  # 清理到比上限少5个，留出缓冲
                excess_count = total_models - target_count

                if excess_count > 0 and idle_models:
                    models_to_remove = idle_models[:min(excess_count, len(idle_models))]
                    removal_reason = f"count_exceed_{excess_count}_over"

            elif force_reason == 'memory_exceed':
                # 内存超限：清理最老的空闲模型，直到内存达标
                target_memory = self.max_memory_mb * 0.8  # 清理到80%的内存使用
                excess_memory = current_memory - target_memory

                if excess_memory > 0 and idle_models:
                    # 预估清理效果：每个模型约释放100MB
                    models_needed = min(len(idle_models), int(excess_memory / self.model_memory_estimate) + 1)
                    models_to_remove = idle_models[:models_needed]
                    removal_reason = f"memory_exceed_{excess_memory:.0f}MB_over"

            elif force_reason == 'manual':
                # 手动清理：清理所有超时空闲模型
                for model_info in idle_models:
                    if model_info['idle_time'] > self.model_timeout:
                        models_to_remove.append(model_info)
                removal_reason = "manual_cleanup"

            else:
                # 正常清理：只清理超时的空闲模型
                for model_info in idle_models:
                    if model_info['idle_time'] > self.model_timeout:
                        models_to_remove.append(model_info)
                removal_reason = "normal_timeout"

            # 🎯 执行清理
            removed_count = 0
            for model_info in models_to_remove:
                model_id = model_info['model_id']
                if self.destroy_model(model_id, force=True):
                    removed_count += 1
                    logger_chatflow.info(f"🧹 清理模型: {model_id}, "
                                         f"创建时间: {model_info['created_time'].strftime('%Y-%m-%d %H:%M:%S')}, "
                                         f"空闲: {model_info['idle_time']:.0f}秒, "
                                         f"原因: {removal_reason}")

            # 🎯 清理后统计
            if removed_count > 0:
                after_memory = self._get_memory_usage()
                memory_saved = current_memory - after_memory
                logger_chatflow.info(f"✅ 清理完成: 移除了 {removed_count} 个模型, "
                                     f"释放内存: {memory_saved:.1f}MB, "
                                     f"剩余模型: {len(self.models)}")

                # 🎯 记录清理效果数据（用于优化配置）
                self._record_cleanup_stats({
                    'before_memory': current_memory,
                    'after_memory': after_memory,
                    'memory_saved': memory_saved,
                    'models_removed': removed_count,
                    'reason': removal_reason,
                    'timestamp': datetime.now().isoformat()
                })

            return {
                'removed_count': removed_count,
                'reason': removal_reason,
                'stats_before': stats,
                'stats_after': {
                    'total_models': len(self.models),
                    'current_memory_mb': self._get_memory_usage()
                }
            }

    def check_and_cleanup_if_needed(self):
        """检查并触发必要的清理 - 🎯 确保这个方法被正确调用"""
        current_memory = self._get_memory_usage()
        total_models = len(self.models)

        force_reason = None

        # 🎯 内存超限检查
        if current_memory > self.max_memory_mb:
            logger_chatflow.warning(f"🚨 内存超限: {current_memory:.1f}MB > {self.max_memory_mb}MB")
            force_reason = 'memory_exceed'

        # 🎯 模型数量超限检查
        elif total_models > self.max_models:
            logger_chatflow.warning(f"⚠️ 模型数量超限: {total_models} > {self.max_models}")
            force_reason = 'count_exceed'

        if force_reason:
            return self.cleanup_idle_models(force_reason=force_reason)

        return {'cleaned': False, 'reason': 'not_needed'}

    def _record_cleanup_stats(self, stats):
        """记录清理统计信息，用于优化配置"""
        # 这里可以存储到文件或数据库，用于分析内存使用模式
        logger_chatflow.debug(f"📈 清理统计: {json.dumps(stats, default=str)}")

    def _build_chatflow_config(self, config_data):
        """根据配置数据构建chatflow配置"""
        # 构建chatflow实例 范本 根据这个样子实例化
        # chatflow_data = {
        #     "agent_config": config_data.agent_config,
        #     "keyword_json": config_data.keyword_json,
        #     "semantic_json": config_data.semantic_json,
        #     "llm_json": config_data.llm_json,
        #     "db_path": DB_PATH,
        #     "design_json": config_data.design_json,
        # }
        # chatflow = build_chatflow(chatflow_data)

        # 这里根据实际的config_data构建配置
        if config_data and 'agent_config' in config_data:
            # 使用自定义配置
            # chatflow_config = ChatFlowConfig.from_custom_data(
            #     agent_config=config_data.get('agent_config'),
            #     keyword_json=config_data.get('keyword_json'),
            #     semantic_json=config_data.get('semantic_json'),
            #     llm_json=config_data.get('llm_json'),
            #     design_json=config_data.get('design_json')
            # )
            # 测试先用默认的 到时候改成实际的
            chatflow_config = ChatFlowConfig.from_paths(
                AGENT_DATA_PATH,
                INTENTION_PATH,
                KNOWLEDGE_PATH,
                DB_PATH,
                CHATFLOW_DESIGN_PATH,
                DIALOG_PATH
            )
        else:
            # 使用默认路径配置
            chatflow_config = ChatFlowConfig.from_paths(
                AGENT_DATA_PATH,
                INTENTION_PATH,
                KNOWLEDGE_PATH,
                DB_PATH,
                CHATFLOW_DESIGN_PATH,
                DIALOG_PATH
            )

        return chatflow_config

    def get_model_status(self, model_id=None):
        """获取模型状态 - 修复JSON序列化问题"""
        with self.lock:
            current_memory = self._get_memory_usage()
            if model_id:
                if model_id in self.models:
                    model_data = self.models[model_id]
                    # 🎯 修复：将 set 转换为 list 以便 JSON 序列化
                    associated_tasks = list(self.model_tasks[model_id]) if model_id in self.model_tasks else []

                    return {
                        'model_id': model_id,
                        'status': 'active',
                        'created_time': model_data['created_time'].isoformat(),
                        'last_used': self.model_last_used[model_id].isoformat(),
                        'usage_count': self.model_usage[model_id],
                        'associated_tasks': associated_tasks,  # 🎯 使用 list 而不是 set
                        'memory_usage': model_data['memory_usage']
                    }
                else:
                    return {'model_id': model_id, 'status': 'not_found'}
            else:
                # 全局状态 - 增强信息
                idle_models = [m for m in self.models if self.model_usage[m] == 0]
                active_models = [m for m in self.models if self.model_usage[m] > 0]

                # 计算内存使用统计
                total_estimated_memory = len(self.models) * self.model_memory_estimate

                # 🎯 修复：构建可序列化的模型状态字典
                models_status = {}
                for model_id, data in self.models.items():
                    # 🎯 修复：将 set 转换为 list
                    associated_tasks_list = list(self.model_tasks[model_id]) if model_id in self.model_tasks else []

                    models_status[model_id] = {
                        'created_time': data['created_time'].isoformat(),
                        'last_used': self.model_last_used[model_id].isoformat(),
                        'usage_count': self.model_usage[model_id],
                        'associated_tasks': associated_tasks_list,  # 🎯 使用 list
                        'status': 'active' if self.model_usage[model_id] > 0 else 'idle',
                        'idle_seconds': (datetime.now() - self.model_last_used[
                            model_id]).total_seconds() if model_id in self.model_last_used else 0
                    }

                return {
                    'total_models': len(self.models),
                    'active_models': len(active_models),
                    'idle_models': len(idle_models),
                    'current_memory_mb': current_memory,
                    'max_memory_mb': self.max_memory_mb,
                    'max_models': self.max_models,
                    'memory_usage_percent': (current_memory / self.max_memory_mb) * 100,
                    'model_usage_percent': (len(self.models) / self.max_models) * 100,
                    'estimated_model_memory_mb': total_estimated_memory,
                    'models': models_status  # 🎯 使用修复后的字典
                }

    def _get_memory_usage(self):
        """获取内存使用情况"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    def check_memory_usage(self):
        """检查内存使用情况，防止内存泄漏"""
        current_memory = self._get_memory_usage()
        # 这里可以保留作为独立的内存检查，但清理逻辑已经集成到智能清理中
        if current_memory > 1024:  # 超过1GB
            logger_chatflow.warning(f"⚠️ 内存使用较高: {current_memory:.1f}MB")
            # 触发智能清理
            self.check_and_cleanup_if_needed()

        return current_memory

    # 🎯 新增：磁盘监控方法
    def check_disk_usage(self):
        """检查磁盘使用情况，防止磁盘爆满"""
        disk_info = self.persistence_manager.get_disk_usage()
        if disk_info and disk_info.get('usage_percent', 0) > 90:
            logger_chatflow.warning(f"⚠️ 磁盘使用率过高: {disk_info['usage_percent']:.1f}%")
            return False
        return True

    # 🎯 新增：手动备份接口
    def create_manual_backup(self):
        """手动创建备份"""
        return self.persistence_manager.create_manual_backup()


# 全局动态模型管理器
model_manager = DynamicModelManager()


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查 - 增强版本"""
    status = model_manager.get_model_status()

    health_status = 'healthy'
    warnings = []

    # 🎯 健康检查逻辑
    if status['memory_usage_percent'] > 90:
        health_status = 'warning'
        warnings.append(f"内存使用率过高: {status['memory_usage_percent']:.1f}%")

    if status['model_usage_percent'] > 90:
        health_status = 'warning'
        warnings.append(f"模型数量接近上限: {status['model_usage_percent']:.1f}%")

    return jsonify({
        'status': health_status,
        'service': 'dynamic_ai_service',
        'timestamp': datetime.now().isoformat(),
        'model_stats': status,
        'warnings': warnings,
        'memory_usage': status['current_memory_mb']
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
    """生成话术接口  加兜底模型逻辑 ，任务id里包含一个model_id 和兜底的model_id 失败的话用兜底在判断一遍"""
    data = request.json
    model_id = data.get('model_id')
    backstop_model = data.get('backstop_model')
    user_input = data.get('user_input', '')
    conversation_history = data.get('conversation_history', [])
    call_id = data.get('call_id', 'unknown')
    task_id = data.get('task_id')

    if not model_id:
        return jsonify({
            'success': False,
            'message': 'model_id 参数不能为空'
        }), 400
    # 🎯 记录实际使用的模型ID
    actual_used_model = model_id
    # 获取模型实例
    chatflow = model_manager.get_model(model_id, task_id)
    if not chatflow:
        # 不存在使用兜底模型，虽然model_id 和backstop_model可能是一个不影响，多判断一次的事儿
        actual_used_model = backstop_model  # 🎯 更新实际使用的模型
        chatflow = model_manager.get_model(backstop_model, task_id)
        if not chatflow:
            # 模型未找到或已过期，通知PHP暂停任务
            if task_id:
                model_manager.notify_php_task_pause(task_id, model_id, "model_not_found_or_expired")

            return jsonify({
                'success': False,
                'message': f'模型 {model_id} 和兜底模型 {backstop_model} 都未找到或已过期',
                'error_code': 'MODEL_NOT_FOUND'
            }), 404

    try:
        print(conversation_history, 'conversation_history')
        # 构建对话状态
        state = ChatState(
            messages=conversation_history.copy(),
            dialog_state=[],
            metadata=[{
                "role":"",
                "content":"",
                "intention_tag":"",
                "dialog_id":"",
                "logic":{
                    "user_logic_title":{},
                    "assistant_logic_title":"",
                    "detail":{}
                }
            }]
        )

        # 添加用户输入
        if user_input and user_input.strip():
            state["messages"].append({"role": "user", "content": user_input})

        # 配置
        conv_config = {"configurable": {"thread_id": f"call_{call_id}"}}
        print(state, '生成话术的请求参数')
        state = chatflow.invoke(state, config=conv_config)
        print(state, 'ai回复详情')
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
        # state["message"] = [
        #   {},{},{}, -- 历史对话
        #   {
        #       "role": "assistant", 角色
        #       "content": "", 回复话术
        # state["metadata"] = [
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
        ai_response = ''
        last_msg = state["messages"][-1]
        if last_msg["role"] == "assistant":
            logger_chatflow.info("智能客服：%s", {last_msg['content']})
            ai_response = last_msg["content"]
            print(f"智能客服：{last_msg['content']}")

        return jsonify({
            'success': True,
            # 当前回复的话术内容
            'response': ai_response,
            # 历史记录
            'conversation_history': state["messages"],
            'conversation_history_detail': state["metadata"],
            'call_id': call_id,
            'model_id': actual_used_model,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger_chatflow.error(f"生成话术失败 - 模型: {actual_used_model}, 呼叫: {call_id}, 错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'话术生成失败: {str(e)}'
        }), 500
    finally:
        # 释放模型使用计数
        if task_id:
            model_manager.release_model(actual_used_model, task_id)


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
    """手动触发清理空闲模型 - 🎯 修复这里"""
    data = request.json
    force = data.get('force', False)

    # 🎯 根据force参数决定清理策略
    if force:
        result = model_manager.cleanup_idle_models(force_reason='manual')
    else:
        result = model_manager.cleanup_idle_models()  # 正常清理

    status = model_manager.get_model_status()
    return jsonify({
        'success': True,
        'message': '空闲模型清理完成',
        'cleanup_result': result,  # 🎯 返回清理详情
        'current_stats': {
            'total_models': status['total_models'],
            'active_models': status['active_models'],
            'current_memory_mb': status['current_memory_mb']
        }
    })


@app.route('/model/persistence/backup', methods=['POST'])
def create_manual_backup():
    """手动创建持久化数据备份"""
    try:
        if model_manager.create_manual_backup():
            return jsonify({
                'success': True,
                'message': '手动备份创建成功'
            })
        else:
            return jsonify({
                'success': False,
                'message': '手动备份创建失败'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'创建备份异常: {str(e)}'
        }), 500


@app.route('/model/persistence/status', methods=['GET'])
def get_persistence_status():
    """获取持久化状态"""
    try:
        disk_info = model_manager.persistence_manager.get_disk_usage()
        models_count = len([f for f in os.listdir(model_manager.persistence_manager.models_dir)
                            if f.endswith('.json')])
        backups_count = len([f for f in os.listdir(model_manager.persistence_manager.backup_dir)
                             if f.endswith('.json')])

        return jsonify({
            'success': True,
            'persistence_path': model_manager.persistence_manager.base_path,
            'models_persisted': models_count,
            'backups_count': backups_count,
            'disk_usage': disk_info,
            'warnings': ['磁盘使用率过高'] if disk_info.get('usage_percent', 0) > 90 else []
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取持久化状态失败: {str(e)}'
        }), 500


def start_dynamic_service(port=5002):
    """启动动态模型服务"""
    logger_chatflow.info(f"启动动态AI模型服务，端口: {port}")
    logger_chatflow.info("服务特点: 动态模型管理，按需创建，自动清理")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


if __name__ == '__main__':
    start_dynamic_service()
