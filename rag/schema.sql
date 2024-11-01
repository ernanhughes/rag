DROP TABLE IF EXISTS CHAT_RESPONSE;
CREATE TABLE CHAT_RESPONSE(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
	model TEXT NOT NULL,
	message_role TEXT,
	message_content TEXT,
	done_reason TEXT,
	done INTEGER,
	total_duration INTEGER,
	load_duration INTEGER,
	prompt_eval_count INTEGER,
	prompt_eval_duration INTEGER,
	eval_count INTEGER,
	eval_duration INTEGER,
	created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

DROP TABLE IF EXISTS CHAT_REQUEST;
CREATE TABLE CHAT_REQUEST(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
	model TEXT NOT NULL,
	message_role TEXT,
	message_content TEXT,
	created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

DROP TABLE IF EXISTS DOCUMENT;
CREATE TABLE DOCUMENT (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
	file_path TEXT NOT NULL,
	file_hash TEXT NOT NULL,
	status TEXT NOT NULL DEFAULT 'pending',
	created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
	completed TIMESTAMP 
);

DROP TABLE IF EXISTS DOCUMENT_FULL_TEXT;
CREATE TABLE DOCUMENT_FULL_TEXT(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
	document_id INTEGER NOT NULL,
	data TEXT,
	created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);


DROP TABLE IF EXISTS DOCUMENT_TEXT_CHUNK;
CREATE TABLE DOCUMENT_TEXT_CHUNK(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
	document_id INTEGER NOT NULL,
	data TEXT,
	created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

DROP TABLE IF EXISTS DOCUMENT_TEXT_CHUNK_FTS;
CREATE VIRTUAL TABLE DOCUMENT_TEXT_CHUNK_FTS USING fts5 (
    chunk_id, data
);

DROP TABLE IF EXISTS DOCUMENT_TEXT_CHUNK_VECTOR;
CREATE VIRTUAL TABLE DOCUMENT_TEXT_CHUNK_VECTOR 
USING vec0(embedding float[1024])