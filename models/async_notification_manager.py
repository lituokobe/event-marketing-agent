# ai_service.py
import requests
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time
import json
from functionals.log_utils import logger_chatflow


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
