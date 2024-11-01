import logging
import os
import sqlite3
from sqlite3 import connect
from typing import Sequence

import sqlite_vec
from sqlite_vec import serialize_float32

from rag._config import appConfig
from rag._utils import compute_file_hash

logger = logging.getLogger(__name__)


class RagDb:
    def __init__(self, db_file: str = appConfig.get("DATABASE_PATH")):
        super().__init__()
        self.db_file = db_file
        self.cn = connect(self.db_file)
        self.cur = self.cn.cursor()
        self.cn.enable_load_extension(True)
        sqlite_vec.load(self.cn)
        self.cn.enable_load_extension(False)
        logger.info(f"Sqlite3 version: {sqlite3.sqlite_version} sqlite_vec version: {self.version()}")

    def insert_document(self, file_path: str):
        file_hash = compute_file_hash(file_path)
        self.cur.execute(
            "INSERT INTO DOCUMENT(file_path, file_hash) VALUES (:1,:2) RETURNING id",
            (file_path, file_hash),
        )
        row = self.cur.fetchone()
        self.cn.commit()
        logger.debug(f"Inserted file {file_path} => {row[0]}")
        return row[0]

    def contains_document(self, file_path: str):
        file_hash = compute_file_hash(file_path)
        self.cur.execute(
            "SELECT 1 FROM DOCUMENT WHERE file_hash = ? AND status = 'INGESTED'",
            (file_hash,)
        )
        row = self.cur.fetchone()
        result = row is not None
        logger.debug(f"File {file_path} hash {file_hash} exists in the database: {result}")
        return result

    def insert_document_text(self, id: str, data: str):
        self.cur.execute(
            "INSERT INTO DOCUMENT_TEXT_CHUNK(document_id, data, text_length) VALUES (:1,:2,:3) RETURNING id",
            (id, data, len(data)),
        )
        row = self.cur.fetchone()
        self.cn.commit()
        logger.info(f"Inserted document text (length {len(data)}) for {id} => {row[0]}")
        return row[0]
    
    def insert_embedding(self, id: str, embedding: str):
        self.cur.execute(
            "INSERT INTO DOCUMENT_TEXT_CHUNK_VECTOR(document_text_id, embedding) VALUES (:1,:2) RETURNING id",
            (id, serialize_float32(embedding)),
        )
        row = self.cur.fetchone()
        self.cn.commit()
        logger.info(f"Inserted embedding for {id} => {row[0]}")
        return row[0]

    def insert_chat_response(self, response: dict):
        sql = """
            INSERT INTO CHAT_RESPONSE (
                model, message_role, message_content, done_reason,
                done, total_duration, load_duration, prompt_eval_count,
                prompt_eval_duration, eval_count, eval_duration
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data = (
            response["model"],
            response["message"]["role"],
            response["message"]["content"],
            response["done_reason"],
            response["done"],
            response["total_duration"],
            response["load_duration"],
            response["prompt_eval_count"],
            response["prompt_eval_duration"],
            response["eval_count"],
            response["eval_duration"],
        )
        try:
            self.cur.execute(sql, data)
            self.cn.commit()
            logger.debug("Video data inserted successfully.")
        except sqlite3.Error as e:
            logger.debug(f"An error occurred: {e}")
            self.cn.rollback()

    @staticmethod
    def init_db(
            db: str = appConfig.get("DATABASE_PATH"),
            schema: str = appConfig.get("SCHEMA_FILE"),
    ):
        logger.info("Initializing the database.....")
        base_dir = os.path.abspath(os.path.dirname(__file__))
        schema_path = os.path.join(base_dir, schema)
        logger.info(f"Db path: {db}")
        logger.info(f"Schema path: {schema_path}")
        rag_db = RagDb(db)
        with open(schema_path, "r") as f:
            schema_sql = f.read()
            logger.info(schema_sql)
            rag_db.cur.executescript(schema_sql)
        rag_db.cn.commit()
        rag_db.cn.close()
        logger.info("Initialized the database")

    def insert_document_embedding(self, id: int, embedding: Sequence[float]):
        logger.info(f"Inserting embedding for {id}")
        self.cur.execute(
            "INSERT INTO DOCUMENT_TEXT_CHUNK_VECTOR(rowid, embedding) VALUES (:1,:2)",
            (id, serialize_float32(list(embedding))),
        )
        self.cn.commit()
        logger.info(f"Inserted document embedding for {id}")


    @staticmethod
    def is_sqlite3_db(filename):
        from os.path import isfile, getsize

        if not isfile(filename):
            return False
        if getsize(filename) < 100:  # SQLite database file header is 100 bytes
            return False

        with open(filename, "rb") as fd:
            header = fd.read(100)

        return header[:16] == b"SQLite format 3\x00"

    def drop_db(self):
        if self.cn:
            self.cn.close()
        self.remove_file(self.db_file)

    @staticmethod
    def remove_file(db):
        if os.path.isfile(db):
            os.remove(db)
            logger.debug(f"Dropped the database:{os.path.abspath(db)}.")
        else:
            logger.debug(f"Database {os.path.abspath(db)} not found.")

    def version(self):
        (vec_version,) = self.cur.execute("select vec_version()").fetchone()
        return vec_version

    def reciprocal_rank_fusion(self, fts_results, vec_results, k=60):  
        rank_dict = {}  
        for rank, (id,) in enumerate(fts_results):  
            if id not in rank_dict:  
                rank_dict[id] = 0  
            rank_dict[id] += 1 / (k + rank + 1)  
        for rank, (rowid, distance) in enumerate(vec_results):  
            if rowid not in rank_dict:  
                rank_dict[rowid] = 0  
            rank_dict[rowid] += 1 / (k + rank + 1)  
    
        # Sort by RRF score  
        sorted_results = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)  
        return sorted_results 