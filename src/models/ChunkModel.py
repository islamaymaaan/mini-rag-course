from bson.objectid import ObjectId
from pymongo import InsertOne

from .BaseDataModel import BaseDataModel
from .db_schemes import DataChunk
from .enums.DataBaseEnum import DataBaseEnum


class ChunckModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client)
        self.collection = self.db_client[
            DataBaseEnum.COLLECTION_CHUNK_NAME.value
        ]
#important
    @classmethod
    async def create_instance(cls, db_client : object):
        instance=cls(db_client)
        await instance.init_collection()
        return instance


    async def init_collection(self):
            all_collections= await self.db_client.list_collection_names()
            if DataBaseEnum.COLLECTION_CHUNK_NAME.value not in all_collections:
                        self.collection = self.db_client[DataBaseEnum.COLLECTION_CHUNK_NAME.value]
                        indexes = DataChunk.get_indexes()
                        for index in indexes:
                            await self.collection.create_index(
                                index["key"],
                                name=index["name"],
                                unique=index["unique"]
                            )

    async def creat_chunk(self, chunk: DataChunk):
        result = await self.collection.insert_one(
            chunk.dict(by_alias=True, exclude_unset=True)
        )
        chunk._id = result.inserted_id
        return chunk

    async def get_chunk(self, chunk_id: str):
        result = await self.collection.find_one(
            {"_id": ObjectId(chunk_id) if isinstance(chunk_id, str) else chunk_id}
        )

        if result is None:
            return None

        return DataChunk(**result)

    async def insert_many_chunks(self, chuncks: list, batch_size: int = 100):
        if not chuncks:
            return 0

        # 🟢 تصليح الـ Step جوه الـ range لتجنب تكرار البيانات
        for i in range(0, len(chuncks), batch_size):
            batch = chuncks[i : i + batch_size]

            operations = [
                InsertOne(chunk.dict(by_alias=True, exclude_unset=True))
                for chunk in batch
            ]

            await self.collection.bulk_write(operations)

        return len(chuncks)

    async def delete_chunks_by_project_id(self, project_id: str):
        # 🟢 تحويل project_id لـ ObjectId لو كان String ينفع يتحول
        target_project_id = project_id
        if isinstance(project_id, str) and ObjectId.is_valid(project_id):
            target_project_id = ObjectId(project_id)

        # تحسباً لو الحقل متخزن كـ ObjectId أو String
        result = await self.collection.delete_many(
            {
                "$or": [
                    {"chunk_project_id": target_project_id},
                    {"chunk_project_id": str(project_id)},
                ]
            }
        )

        return result.deleted_count

    async def get_poject_chunks(self, project_id: ObjectId, page_no: int=1, page_size: int=50):
        records = await self.collection.find({
                    "chunk_project_id": project_id
                }).skip(
                    (page_no-1) * page_size
                ).limit(page_size).to_list(length=None)

        return [
            DataChunk(**record)
            for record in records
        ]