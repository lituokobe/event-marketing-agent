import hashlib
from pymilvus import MilvusClient, MilvusException
from pymilvus.milvus_client import IndexParams
from functionals.log_utils import logger_chatflow
from models.embedding_functions import embed_query
# from models.embeddings import qwen3_embedding_model

class LaunchMilvus:
    def __init__(self, vector_db_url: str, collection_name: str, intentions: list = None, knowledge: list = None):
        self.client = MilvusClient(uri = vector_db_url, secure=False)
        self.collection_name = collection_name
        self.merged_data = (intentions or []) + (knowledge or [])
        self._ensure_collection_ready(self.merged_data)

    def _generate_phrase_id(self, intention_id: str, phrase: str) -> int:
        """Generate a deterministic ID for a phrase."""
        hash_str = hashlib.sha256(f"{intention_id}:{phrase}".encode()).hexdigest()
        # Convert to int and mask to 63 bits (safe for INT64)
        return int(hash_str[:16], 16) & ((1 << 63) - 1)

    def _ensure_collection_ready(self, merged_data: list = None):
        """Ensure collection exists and is ready with data."""
        # if collection doesn't exist, create it.
        if not self.client.has_collection(self.collection_name):
            self._create_collection()
            self._create_hnsw_index()
            logger_chatflow.info(f"已创建新的 collection '{self.collection_name}' 和 HNSW index")
        else:
            self._ensure_hnsw_index()
            logger_chatflow.info(f"Collection '{self.collection_name}' 已存在，使用已有的 HNSW index")

        # Always upsert latest data
        if merged_data:
            self._upsert_intention_data(merged_data)

        # Ensure the collection is loaded
        self.client.load_collection(self.collection_name, timeout=30)

    def _create_collection(self):
        """Create the Milvus collection schema."""
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension = 1024, #This value has to be 1024 as it is the dimension of Qwen Embedding
            metric_type="COSINE",
            auto_id=False,  # We manage IDs by ourselves
            primary_field="id",
            extra_fields=[
                {"name": "id", "data_type": "INT64", "is_primary": True},
                {"name": "intention_id", "data_type": "VARCHAR", "max_length": 512},
                {"name": "intention_name", "data_type": "VARCHAR", "max_length": 512},
                {"name": "phrase", "data_type": "VARCHAR", "max_length": 4096},
            ]
        )
        logger_chatflow.info(f"创建向量数据库collection: {self.collection_name}")

    def _create_hnsw_index(self):
        """Create HNSW index - called only once when collection is created."""
        try:
            # First, drop any existing indexes on the vector field
            existing_indexes = self.client.list_indexes(self.collection_name)
            for index_name in existing_indexes:
                try:
                    self.client.release_collection(self.collection_name)
                    self.client.drop_index(self.collection_name, index_name)
                    logger_chatflow.info(f"已删除现有index: {index_name}")
                except Exception as e:
                    logger_chatflow.warning(f"删除index {index_name} 时发生警告: {str(e)}")
            # Create HNSW index
            index_params = IndexParams()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200}
            )
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            logger_chatflow.info(f"创建向量数据库HNSW index")
        except MilvusException as e:
            logger_chatflow.info(f"创建 HNSW index时发生错误: {str(e)}")
            raise RuntimeError(str(e))

    def _ensure_hnsw_index(self):
        """Ensure HNSW index exists, create if missing or wrong type."""
        try:
            self.client.release_collection(self.collection_name)
            existing_indexes = self.client.list_indexes(self.collection_name)

            if not existing_indexes:
                # No index exists, create HNSW
                self._create_hnsw_index()
                return

            # Check if HNSW index exists
            hnsw_exists = False
            for index_name in existing_indexes:
                try:
                    index_info = self.client.describe_index(self.collection_name, index_name)
                    if (index_info.get('index_type') == 'HNSW' and
                            index_info.get('metric_type') == 'COSINE'):
                        hnsw_exists = True
                        logger_chatflow.info(f"HNSW index 已存在，index名称为 {index_name}")
                        break
                except Exception as e:
                    logger_chatflow.warning(f"检查index {index_name} 时发生警告: {str(e)}")

            if not hnsw_exists:
                # Existing index is not HNSW, replace it
                logger_chatflow.info("现有index不是 HNSW 类型，替换为 HNSW")
                self._create_hnsw_index()

        except Exception as e:
            logger_chatflow.error(f"检查 HNSW index时发生错误: {str(e)}")
            raise RuntimeError(str(e))

    def _upsert_intention_data(self, merged_data: list):
        """Upsert intention data - index automatically handles new vectors."""
        upsert_data = []
        for item in merged_data:
            intention_id = item.get("intention_id")
            intention_name = item.get("intention_name")
            for phrase in item.get("semantic", []):
                if phrase.strip():  # skip empty
                    # embedding = qwen3_embedding_model.embed_documents(phrase)
                    embedding = embed_query(phrase)
                    if hasattr(embedding, 'tolist'):#convert array-like objects into standard lists, required by Milvus
                        embedding = embedding.tolist()
                    if len(embedding)!=1024:
                        e_m = f"向量为度应为1024，目前为{len(embedding)}"
                        logger_chatflow.error(e_m)
                        raise ValueError(e_m)

                    phrase_id = self._generate_phrase_id(intention_id, phrase)
                    upsert_data.append({
                        "id": phrase_id,
                        "vector": embedding,
                        "intention_id": intention_id,
                        "intention_name": intention_name,
                        "phrase": phrase
                    })

        if not upsert_data:
            logger_chatflow.info(f"没有问法短语插入向量数据库")
            return

        # Upsert (insert or replace)
        self.client.upsert(collection_name=self.collection_name, data=upsert_data)
        logger_chatflow.info(f"更新 {len(upsert_data)} 条问法短语到向量数据库")

    def get_index_info(self):
        """Get information about current index."""
        try:
            existing_indexes = self.client.list_indexes(self.collection_name)
            index_info_list = []
            for index_name in existing_indexes:
                index_info = self.client.describe_index(self.collection_name, index_name)
                index_info_list.append({
                    'name': index_name,
                    'info': index_info
                })
            return index_info_list
        except Exception as e:
            logger_chatflow.error(f"获取index信息时发生错误: {str(e)}")
            return []