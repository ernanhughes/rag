import io
import sqlite3
from typing import List, Tuple, Any

import numpy as np


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597
    """
    out = io.BytesIO()
    np.save(out, arr)  # noqa
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    """
    https://stackoverflow.com/a/18622264
    """
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)  # noqa


def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points represented as NumPy arrays.
    """
    if point1.shape != point2.shape:
        raise ValueError("Input points must have the same shape.")

    # Calculate the Euclidean distance
    distance = np.linalg.norm(point1 - point2)

    return distance


def get_nearest_neighbor(train, test_row, num_neighbors: int = 1):
    """
    Find the nearest neighbors of a test data point in a dataset.
    """
    distances = []

    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))

    distances.sort(key=lambda tup: tup[1])
    neighbors = []

    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


class SQLiteDB:
    def __init__(self, database: str = ":memory:"):
        self.conn = sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cur = self.conn.cursor()

    def _create_table(self, table_name: str, columns: List[Tuple[str, str]]):
        """
        Create a table with the given name and columns

        Columns should be a list of tuples (name, type)

        For example, [("id", "INTEGER PRIMARY KEY"), ("name", "TEXT")]
        """
        sql = f"CREATE TABLE {table_name} ("
        for column in columns:
            sql += f"{column[0]} {column[1]}, "
        sql = sql[:-2] + ")"  # Removes the last comma and add a closing parenthesis

        self.cur.execute(sql)
        self.conn.commit()

    def _insert_data(self, table_name: str, data: List[Tuple[Any]]):
        """
        Insert data into the table

        Data should be a list of tuples with values for each column

        For example, [(1, "Alice"), (2, "Bob")]
        """
        placeholders = ", ".join(
            ["?"] * len(data[0])
        )  # Create placeholders for each value
        sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
        self.cur.executemany(sql, data)
        self.conn.commit()

    def _query_data(self, table_name: str, condition: str = None) -> List[Tuple]:
        """
        Query data from the table

        Condition is an optional string to filter the results

        For example, "name = 'Alice'"
        """
        sql = f"SELECT * FROM {table_name}"
        if condition:
            sql += f" WHERE {condition}"
        self.cur.execute(sql)
        return self.cur.fetchall()  # Return a list of tuples with the query results

    def _close(self):
        """
        Create placeholders for each value
        """
        self.conn.close()


class VectorDB(SQLiteDB):
    def __init__(self, collection_name: str):
        """
        Initialize a VectorDB instance for storing and querying vectors.
        """
        super().__init__()
        self.collection_name = collection_name

    def create(self):
        """
        Create the collection (table) for storing vectors with the name specified during initialization.
        """
        columns = [("arr", "array")]
        self._create_table(self.collection_name, columns)

    def insert(self, vectors: List[np.array]):
        """
        Insert a list of vectors (NumPy arrays) into the collection.
        """
        _vectors = [(vector,) for vector in vectors]
        self._insert_data(self.collection_name, _vectors)

    def search(self, query: np.array, num_results: int):
        """
        Find the nearest neighbors of a query vector in the collection.
        """
        vectors = self._query_data(self.collection_name)
        vectors = [vector[0] for vector in vectors]
        return get_nearest_neighbor(vectors, query, num_results)