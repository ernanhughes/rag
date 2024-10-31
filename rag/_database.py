import os
import sqlite3
from dataclasses import asdict
from sqlite3 import connect
import sqlite_vec
from rag._config import appConfig
from rag._utils import compute_file_hash

import logging

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

    def insert_document(self, file_path: str):
        file_hash = compute_file_hash(file_path)
        self.cur.execute(
            "INSERT INTO DOCUMENT(file_path, file_hash) VALUES (:1,:2)",
            (file_path, file_hash),
        )
        self.cn.commit()
        logger.debug(f"Inserted file {file_path}")
        return self.cur.lastrowid

    def insert_document_text(self, id: str, data: str):
        self.cur.execute(
            "INSERT INTO DOCUMENT_TEXT(document_id, data) VALUES (:1,:2)",
            (id, data),
        )
        self.cn.commit()
        logger.info(f"Inserted document text (length {len(data)}) for {id}")


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
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        with open(schema_path, "r") as f:
            schema_sql = f.read()
            logger.info(schema_sql)
            cursor.executescript(schema_sql)
        conn.commit()
        conn.close()
        logger.info("Initialized the database")
        rag_db = RagDb()
        logger.info(f"Vector Database (sqlite-vec)=> vec_version={rag_db.version()}")

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
