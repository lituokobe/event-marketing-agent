# persistence_manager.py
import json
import os
import shutil
from datetime import datetime, timedelta
import logging
from threading import RLock


class ModelPersistenceManager:
    """模型持久化管理器 - 专注模型配置"""

    def __init__(self, base_path=None):
        # 🎯 使用相对路径，默认为当前目录下的 persistence 文件夹
        if base_path is None:
            # 获取当前文件所在目录的绝对路径，然后构建相对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            current_dir += '/../'
            base_path = os.path.join(current_dir, "persistence")
        print(base_path, 'base_path')
        self.base_path = base_path
        self.models_dir = os.path.join(base_path, "models")
        self.backup_dir = os.path.join(base_path, "../backups")
        self.recovery_log = os.path.join(base_path, "recovery.log")
        self.lock = RLock()

        # 创建目录结构
        self._ensure_directories()
        self._setup_logging()

    def _ensure_directories(self):
        """确保目录结构存在"""
        for directory in [self.models_dir, self.backup_dir]:
            os.makedirs(directory, exist_ok=True)

    def _setup_logging(self):
        """设置恢复日志"""
        self.logger = logging.getLogger('model_persistence')
        handler = logging.FileHandler(self.recovery_log, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def save_model_config(self, model_id, model_data):
        """保存模型配置到文件"""
        with self.lock:
            try:
                file_path = os.path.join(self.models_dir, f"{model_id}.json")

                # 准备持久化数据
                persist_data = {
                    'model_id': model_id,
                    'config': model_data.get('config', {}),
                    'created_time': model_data.get('created_time').isoformat() if model_data.get(
                        'created_time') else None,
                    'expire_time': model_data.get('expire_time'),
                    'memory_usage': model_data.get('memory_usage', 0),
                    'status': model_data.get('status', 'active'),
                    'last_used': model_data.get('last_used').isoformat() if model_data.get('last_used') else None,
                    'persisted_at': datetime.now().isoformat()
                }

                # 写入文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(persist_data, f, ensure_ascii=False, indent=2)

                self.logger.info(f"✅ 模型配置已保存: {model_id}")
                return True

            except Exception as e:
                self.logger.error(f"❌ 保存模型配置失败 {model_id}: {str(e)}")
                return False

    def load_model_configs(self):
        """加载所有模型配置"""
        with self.lock:
            models = {}
            try:
                if not os.path.exists(self.models_dir):
                    return models

                for filename in os.listdir(self.models_dir):
                    if filename.endswith('.json'):
                        model_id = filename[:-5]
                        file_path = os.path.join(self.models_dir, filename)

                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                model_data = json.load(f)

                            # 转换时间字段
                            if model_data.get('created_time'):
                                model_data['created_time'] = datetime.fromisoformat(model_data['created_time'])
                            if model_data.get('last_used'):
                                model_data['last_used'] = datetime.fromisoformat(model_data['last_used'])

                            models[model_id] = model_data
                            self.logger.info(f"📥 加载模型配置: {model_id}")

                        except Exception as e:
                            self.logger.error(f"❌ 加载模型配置失败 {filename}: {str(e)}")
                            continue

            except Exception as e:
                self.logger.error(f"❌ 加载模型配置目录失败: {str(e)}")

            return models

    def delete_model_config(self, model_id):
        """删除模型配置"""
        with self.lock:
            try:
                file_path = os.path.join(self.models_dir, f"{model_id}.json")
                if os.path.exists(file_path):
                    # 🎯 自动备份到备份目录
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_path = os.path.join(self.backup_dir, f"{model_id}_{timestamp}.json")
                    shutil.copy2(file_path, backup_path)

                    # 删除原文件
                    os.remove(file_path)
                    self.logger.info(f"🗑️ 模型配置已删除: {model_id}")

                    # 🎯 自动清理旧备份
                    self._cleanup_old_backups()

                return True
            except Exception as e:
                self.logger.error(f"❌ 删除模型配置失败 {model_id}: {str(e)}")
                return False

    def _cleanup_old_backups(self, keep_days=7):
        """清理旧备份 - 在删除模型时自动调用"""
        try:
            current_time = datetime.now()
            deleted_count = 0

            for filename in os.listdir(self.backup_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.backup_dir, filename)

                    # 获取文件修改时间
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))

                    # 检查是否超过保留期限
                    if (current_time - file_mtime).days > keep_days:
                        os.remove(file_path)
                        deleted_count += 1
                        self.logger.info(f"🧹 清理旧备份: {filename}")

            if deleted_count > 0:
                self.logger.info(f"✅ 自动清理完成: 删除了 {deleted_count} 个旧备份")

        except Exception as e:
            self.logger.error(f"❌ 自动清理备份失败: {str(e)}")

    def create_manual_backup(self):
        """手动创建完整备份"""
        with self.lock:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = os.path.join(self.backup_dir, f"manual_backup_{timestamp}")

                shutil.copytree(self.models_dir, backup_path)
                self.logger.info(f"💾 创建手动备份: {backup_path}")
                return True

            except Exception as e:
                self.logger.error(f"❌ 创建手动备份失败: {str(e)}")
                return False

    def get_disk_usage(self):
        """获取磁盘使用情况"""
        try:
            total, used, free = shutil.disk_usage(self.base_path)
            return {
                'total_mb': total // (1024 * 1024),
                'used_mb': used // (1024 * 1024),
                'free_mb': free // (1024 * 1024),
                'usage_percent': (used / total) * 100
            }
        except Exception as e:
            self.logger.error(f"❌ 获取磁盘使用情况失败: {str(e)}")
            return {}
