# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Literal, Optional

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from psycopg import AsyncConnection, Connection, Error, IntegrityError
from psycopg.abc import Query
from psycopg.connection_async import AsyncCursor
from psycopg.cursor import Cursor
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from psycopg.sql import Literal as SQLLiteral
from psycopg import sql

from pgvector.psycopg import register_vector, register_vector_async

from .converters import _from_haystack_to_pg_documents, _from_pg_to_haystack_documents
from .filters import _convert_filters_to_where_clause_and_params, _validate_filters

logger = logging.getLogger(__name__)

CREATE_TABLE_STATEMENT = """
CREATE TABLE {schema_name}.{table_name} (
id VARCHAR(128) PRIMARY KEY,
embedding VECTOR({embedding_dimension}),
content TEXT,
blob_data BYTEA,
blob_meta JSONB,
blob_mime_type VARCHAR(255),
meta JSONB)
"""

INSERT_STATEMENT = """
INSERT INTO {schema_name}.{table_name}
(id, embedding, content, blob_data, blob_meta, blob_mime_type, meta)
VALUES (%(id)s, %(embedding)s, %(content)s, %(blob_data)s, %(blob_meta)s, %(blob_mime_type)s, %(meta)s)
"""

UPDATE_STATEMENT = """
ON CONFLICT (id) DO UPDATE SET
embedding = EXCLUDED.embedding,
content = EXCLUDED.content,
blob_data = EXCLUDED.blob_data,
blob_meta = EXCLUDED.blob_meta,
blob_mime_type = EXCLUDED.blob_mime_type,
meta = EXCLUDED.meta
"""

KEYWORD_QUERY = """
SELECT {table_name}.*, ts_rank_cd(to_tsvector({language}, content), query) AS score
FROM {schema_name}.{table_name}, plainto_tsquery({language}, %s) query
WHERE to_tsvector({language}, content) @@ query
"""

VALID_VECTOR_FUNCTIONS = ["cosine_similarity", "inner_product", "l2_distance"]

VECTOR_FUNCTION_TO_POSTGRESQL_OPS = {
    "cosine_similarity": "vector_cosine_ops",
    "inner_product": "vector_ip_ops",
    "l2_distance": "vector_l2_ops",
}

HNSW_INDEX_CREATION_VALID_KWARGS = ["m", "ef_construction"]


class PgvectorDocumentStore:
    """
    A Document Store using PostgreSQL with the [pgvector extension](https://github.com/pgvector/pgvector) installed.
    """

    def __init__(
        self,
        *,
        connection_string: Secret = Secret.from_env_var("PG_CONN_STR"),
        create_extension: bool = True,
        schema_name: str = "public",
        table_name: str = "haystack_documents",
        language: str = "english",
        embedding_dimension: int = 1536,
        vector_function: Literal["cosine_similarity", "inner_product", "l2_distance"] = "inner_product",
        recreate_table: bool = False,
        search_strategy: Literal["exact_nearest_neighbor", "hnsw"] = "exact_nearest_neighbor",
        hnsw_recreate_index_if_exists: bool = False,
        hnsw_index_creation_kwargs: Optional[Dict[str, int]] = None,
        hnsw_index_name: str = "haystack_hnsw_index",
        hnsw_ef_search: Optional[int] = None,
        keyword_index_name: str = "haystack_keyword_index",
        metadata_schema: Optional[Dict[str, str]] = None,
        metadata_to_index: Optional[List[str]] = None,
    ):
        """
        Creates a new PgvectorDocumentStore instance.
        It is meant to be connected to a PostgreSQL database with the pgvector extension installed.
        A specific table to store Haystack documents will be created if it doesn't exist yet.

        :param connection_string: The connection string to use to connect to the PostgreSQL database, defined as an
            environment variable. It can be provided in either URI format
            e.g.: `PG_CONN_STR="postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"`, or keyword/value format
            e.g.: `PG_CONN_STR="host=HOST port=PORT dbname=DBNAME user=USER password=PASSWORD"`
            See [PostgreSQL Documentation](https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING)
            for more details.
        :param create_extension: Whether to create the pgvector extension if it doesn't exist.
            Set this to `True` (default) to automatically create the extension if it is missing.
            Creating the extension may require superuser privileges.
            If set to `False`, ensure the extension is already installed; otherwise, an error will be raised.
        :param schema_name: The name of the schema the table is created in. The schema must already exist.
        :param table_name: The name of the table to use to store Haystack documents.
        :param language: The language to be used to parse query and document content in keyword retrieval.
            To see the list of available languages, you can run the following SQL query in your PostgreSQL database:
            `SELECT cfgname FROM pg_ts_config;`.
            More information can be found in this [StackOverflow answer](https://stackoverflow.com/a/39752553).
        :param embedding_dimension: The dimension of the embedding.
        :param vector_function: The similarity function to use when searching for similar embeddings.
            `"cosine_similarity"` and `"inner_product"` are similarity functions and
            higher scores indicate greater similarity between the documents.
            `"l2_distance"` returns the straight-line distance between vectors,
            and the most similar documents are the ones with the smallest score.
            **Important**: when using the `"hnsw"` search strategy, an index will be created that depends on the
            `vector_function` passed here. Make sure subsequent queries will keep using the same
            vector similarity function in order to take advantage of the index.
        :param recreate_table: Whether to recreate the table if it already exists.
        :param search_strategy: The search strategy to use when searching for similar embeddings.
            `"exact_nearest_neighbor"` provides perfect recall but can be slow for large numbers of documents.
            `"hnsw"` is an approximate nearest neighbor search strategy,
            which trades off some accuracy for speed; it is recommended for large numbers of documents.
            **Important**: when using the `"hnsw"` search strategy, an index will be created that depends on the
            `vector_function` passed here. Make sure subsequent queries will keep using the same
            vector similarity function in order to take advantage of the index.
        :param hnsw_recreate_index_if_exists: Whether to recreate the HNSW index if it already exists.
            Only used if search_strategy is set to `"hnsw"`.
        :param hnsw_index_creation_kwargs: Additional keyword arguments to pass to the HNSW index creation.
            Only used if search_strategy is set to `"hnsw"`. You can find the list of valid arguments in the
            [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw)
        :param hnsw_index_name: Index name for the HNSW index.
        :param hnsw_ef_search: The `ef_search` parameter to use at query time. Only used if search_strategy is set to
            `"hnsw"`. You can find more information about this parameter in the
            [pgvector documentation](https://github.com/pgvector/pgvector?tab=readme-ov-file#hnsw).
        :param keyword_index_name: Index name for the Keyword index.
        :param metadata_schema: A dictionary mapping metadata field names to their PostgreSQL types (e.g.,
            `{\"author\": \"TEXT\", \"year\": \"INTEGER\"}`). If provided, these fields will be stored in separate
            columns. If `None` (default), metadata is stored in a single JSONB column named 'meta'.
            Column names derived from metadata keys will be sanitized by replacing non-alphanumeric
            characters with underscores. Ensure provided types are valid PostgreSQL types.
        :param metadata_to_index: A list of metadata field names to index. If provided, only these fields will be indexed.
            If `None` (default), all metadata fields will be indexed.
        """

        self.connection_string = connection_string
        self.create_extension = create_extension
        self.table_name = table_name
        self.schema_name = schema_name
        self.embedding_dimension = embedding_dimension
        if vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}, but got {vector_function}"
            raise ValueError(msg)
        self.vector_function = vector_function
        self.recreate_table = recreate_table
        self.search_strategy = search_strategy
        self.hnsw_recreate_index_if_exists = hnsw_recreate_index_if_exists
        self.hnsw_index_creation_kwargs = hnsw_index_creation_kwargs or {}
        self.hnsw_index_name = hnsw_index_name
        self.hnsw_ef_search = hnsw_ef_search
        self.keyword_index_name = keyword_index_name
        self.language = language
        # Store the original schema before sanitization
        self.original_metadata_schema = metadata_schema
        # Store the sanitized metadata schema (keys are sanitized for SQL column names)
        self.sanitized_metadata_schema = self._sanitize_metadata_schema(metadata_schema) if metadata_schema else None
        self.metadata_to_index = metadata_to_index

        self._connection = None
        self._async_connection = None
        self._cursor = None
        self._async_cursor = None
        self._dict_cursor = None
        self._async_dict_cursor = None
        self._table_initialized = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            connection_string=self.connection_string.to_dict(),
            create_extension=self.create_extension,
            schema_name=self.schema_name,
            table_name=self.table_name,
            embedding_dimension=self.embedding_dimension,
            vector_function=self.vector_function,
            recreate_table=self.recreate_table,
            search_strategy=self.search_strategy,
            hnsw_recreate_index_if_exists=self.hnsw_recreate_index_if_exists,
            hnsw_index_creation_kwargs=self.hnsw_index_creation_kwargs,
            hnsw_index_name=self.hnsw_index_name,
            hnsw_ef_search=self.hnsw_ef_search,
            keyword_index_name=self.keyword_index_name,
            language=self.language,
            # Serialize original metadata_schema (sanitized version is derived)
            metadata_schema=self.original_metadata_schema,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PgvectorDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, ["connection_string"])
        return default_from_dict(cls, data)

    @staticmethod
    def _connection_is_valid(connection):
        """
        Internal method to check if the connection is still valid.
        """

        # implementation inspired to psycopg pool
        # https://github.com/psycopg/psycopg/blob/d38cf7798b0c602ff43dac9f20bbab96237a9c38/psycopg_pool/psycopg_pool/pool.py#L528

        try:
            connection.execute("")
        except Error:
            return False
        return True

    @staticmethod
    async def _connection_is_valid_async(connection):
        """
        Internal method to check if the async connection is still valid.
        """
        try:
            await connection.execute("")
        except Error:
            return False
        return True

    def _execute_sql(
        self, sql_query: Query, params: Optional[tuple] = None, error_msg: str = "", cursor: Optional[Cursor] = None
    ):
        """
        Internal method to execute SQL statements and handle exceptions.

        :param sql_query: The SQL query to execute.
        :param params: The parameters to pass to the SQL query.
        :param error_msg: The error message to use if an exception is raised.
        :param cursor: The cursor to use to execute the SQL query. Defaults to self._cursor.
        """

        params = params or ()
        cursor = cursor or self._cursor

        if cursor is None or self._connection is None:
            message = (
                "The cursor or the connection is not initialized. "
                "Make sure to call _ensure_db_setup() before calling this method."
            )
            raise ValueError(message)

        sql_query_str = sql_query.as_string(cursor) if not isinstance(sql_query, str) else sql_query
        logger.debug("SQL query: {query}\nParameters: {parameters}", query=sql_query_str, parameters=params)

        try:
            result = cursor.execute(sql_query, params)
        except Error as e:
            self._connection.rollback()
            detailed_error_msg = f"{error_msg}.\nYou can find the SQL query and the parameters in the debug logs."
            raise DocumentStoreError(detailed_error_msg) from e

        return result

    async def _execute_sql_async(
        self,
        sql_query: Query,
        params: Optional[tuple] = None,
        error_msg: str = "",
        cursor: Optional[AsyncCursor] = None,
    ):
        """
        Internal method to asynchronously execute SQL statements and handle exceptions.

        :param sql_query: The SQL query to execute.
        :param params: The parameters to pass to the SQL query.
        :param error_msg: The error message to use if an exception is raised.
        :param cursor: The cursor to use to execute the SQL query. Defaults to self._async_cursor.
        """

        params = params or ()
        cursor = cursor or self._async_cursor

        if cursor is None or self._async_connection is None:
            message = (
                "The cursor or the connection is not initialized. "
                "Make sure to call _ensure_db_setup_async() before calling this method."
            )
            raise ValueError(message)

        sql_query_str = sql_query.as_string(cursor) if not isinstance(sql_query, str) else sql_query
        logger.debug("SQL query: {query}\nParameters: {parameters}", query=sql_query_str, parameters=params)

        try:
            result = await cursor.execute(sql_query, params)
        except Error as e:
            await self._async_connection.rollback()
            detailed_error_msg = f"{error_msg}.\nYou can find the SQL query and the parameters in the debug logs."
            raise DocumentStoreError(detailed_error_msg) from e

        return result

    def _ensure_db_setup(self):
        """
        Ensures that the connection to the PostgreSQL database exists and is valid.
        If not, connection and cursors are created.
        If the table is not initialized, it will be set up.
        """
        if self._connection and self._cursor and self._dict_cursor and self._connection_is_valid(self._connection):
            return

        # close the connection if it already exists
        if self._connection:
            try:
                self._connection.close()
            except Error as e:
                logger.debug("Failed to close connection: {e}", e=str(e))

        conn_str = self.connection_string.resolve_value() or ""
        connection = Connection.connect(conn_str)
        connection.autocommit = True
        if self.create_extension:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)  # Note: this must be called before creating the cursors.

        self._connection = connection
        self._cursor = self._connection.cursor()
        self._dict_cursor = self._connection.cursor(row_factory=dict_row)

        if not self._table_initialized:
            self._initialize_table()

    async def _ensure_db_setup_async(self):
        """
        Async internal method.
        Ensures that the connection to the PostgreSQL database exists and is valid.
        If not, connection and cursors are created.
        If the table is not initialized, it will be set up.
        """

        if (
            self._async_connection
            and self._async_cursor
            and self._async_dict_cursor
            and await self._connection_is_valid_async(self._async_connection)
        ):
            return

        # close the connection if it already exists
        if self._async_connection:
            await self._async_connection.close()

        conn_str = self.connection_string.resolve_value() or ""
        async_connection = await AsyncConnection.connect(conn_str)
        await async_connection.set_autocommit(True)
        if self.create_extension:
            await async_connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector_async(async_connection)  # Note: this must be called before creating the cursors.

        self._async_connection = async_connection
        self._async_cursor = self._async_connection.cursor()
        self._async_dict_cursor = self._async_connection.cursor(row_factory=dict_row)

        if not self._table_initialized:
            await self._initialize_table_async()

    def _build_table_creation_queries(self):
        """
        Internal method to build the SQL queries for table creation and indexing.
        Dynamically builds the CREATE TABLE statement based on metadata_schema
        and CREATE INDEX statements for specified metadata fields and the FTS column.
        """

        sql_table_exists = SQL("SELECT 1 FROM pg_tables WHERE schemaname = %s AND tablename = %s")

        # Start building the CREATE TABLE statement dynamically
        column_definitions = [
            SQL("id VARCHAR(128) PRIMARY KEY"),
            SQL("embedding VECTOR({embedding_dimension})").format(embedding_dimension=SQLLiteral(self.embedding_dimension)),
            SQL("content TEXT"),
            # Add the generated tsvector column using coalesce for safety
            # The language parameter is crucial here.
            SQL("content_fts tsvector GENERATED ALWAYS AS (to_tsvector({language}, coalesce(content, ''))) STORED").format(language=SQLLiteral(self.language)),
            SQL("blob_data BYTEA"),
            SQL("blob_meta JSONB"),
            SQL("blob_mime_type VARCHAR(255)"),
        ]

        # Add columns based on metadata_schema if provided
        if self.sanitized_metadata_schema:
            logger.info(f"Using metadata_schema to create flat columns: {self.sanitized_metadata_schema}")
            # Use sanitized keys for column names and assume types are valid SQL
            for sanitized_key, sql_type in self.sanitized_metadata_schema.items():
                # Basic validation/warning for type - could be enhanced
                if not isinstance(sql_type, str) or not sql_type.isalnum():
                     logger.warning(f"Potential issue with SQL type '{sql_type}' for metadata column '{sanitized_key}'. Ensure it's a valid PostgreSQL type.")
                # Use SQL() to safely include the type string, and Identifier for the column name
                column_definitions.append(SQL("{col_name} {col_type}").format(
                    col_name=Identifier(sanitized_key),
                    col_type=SQL(sql_type) # Assume sql_type is a valid SQL type string
                ))
        else:
            # Fallback: If no metadata_schema, add the original JSONB meta column
            logger.info("No metadata_schema provided, using single 'meta' JSONB column.")
            column_definitions.append(SQL("meta JSONB"))

        # Combine column definitions into the final CREATE TABLE statement
        sql_create_table = SQL("CREATE TABLE {schema_name}.{table_name} ({columns})").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            columns=SQL(", ").join(column_definitions)
        )

        # Query to check if a specific index exists
        sql_index_exists = SQL(
            "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND tablename = %s AND indexname = %s"
        )

        # --- Keyword Index (on the new generated column) ---
        keyword_index_name_identifier = Identifier(self.keyword_index_name)
        # Create the GIN index on the pre-generated 'content_fts' column
        # Use CREATE INDEX CONCURRENTLY for non-blocking creation
        sql_create_keyword_index = SQL(
            "CREATE INDEX CONCURRENTLY {index_name} ON {schema_name}.{table_name} USING GIN (content_fts) WITH (fastupdate = off)" # Index the generated column, disable fastupdate for compactness
        ).format(
            schema_name=Identifier(self.schema_name),
            index_name=keyword_index_name_identifier,
            table_name=Identifier(self.table_name),
        )

        # --- Metadata Indices ---
        # Generate CREATE INDEX statements for metadata fields specified in metadata_to_index
        metadata_index_queries = {} # Store as {index_name_identifier: create_sql}
        if self.sanitized_metadata_schema and self.metadata_to_index:
            # Get the sanitized keys corresponding to the original keys in metadata_to_index
            # We need a mapping from original key -> sanitized key
            original_to_sanitized = {k: sk for k, sk in self._sanitize_metadata_schema(self.original_metadata_schema).items()}

            for original_key in self.metadata_to_index:
                sanitized_key = original_to_sanitized.get(original_key)
                if not sanitized_key:
                    logger.warning(f"Metadata key '{original_key}' specified in 'metadata_to_index' not found in 'metadata_schema'. Skipping index creation.")
                    continue
                if sanitized_key not in self.sanitized_metadata_schema:
                    logger.warning(f"Sanitized key '{sanitized_key}' (from original '{original_key}') not found in sanitized schema. Skipping index creation.")
                    continue

                # Generate a predictable index name
                index_name_str = f"idx_{self.table_name}_{sanitized_key}"[:63] # Ensure valid length
                index_name_identifier = Identifier(index_name_str)
                sanitized_key_identifier = Identifier(sanitized_key)

                # Basic index creation, assumes B-tree which is good for most types
                # Consider allowing index type specification in the future
                # Use CREATE INDEX CONCURRENTLY for non-blocking creation
                sql_create_meta_index = SQL(
                    "CREATE INDEX CONCURRENTLY {index_name} ON {schema_name}.{table_name} ({column_name}) WITH (fastupdate = off)"
                ).format(
                    index_name=index_name_identifier,
                    schema_name=Identifier(self.schema_name),
                    table_name=Identifier(self.table_name),
                    column_name=sanitized_key_identifier,
                )
                metadata_index_queries[index_name_identifier] = sql_create_meta_index
                logger.debug(f"Generated CREATE INDEX statement for metadata field: {original_key} (column: {sanitized_key}, index: {index_name_str})")
        elif self.metadata_to_index and not self.sanitized_metadata_schema:
             # Only add index on the 'meta' JSONB column if specified and not using flat schema
             if "meta" in self.metadata_to_index:
                 logger.info("Creating GIN index on the 'meta' JSONB column as requested in 'metadata_to_index'.")
                 # Generate a predictable index name
                 index_name_str = f"idx_{self.table_name}_meta_jsonb"[:63] # Ensure valid length
                 index_name_identifier = Identifier(index_name_str)
                 # Use GIN index for JSONB for efficient querying of keys/values
                 # Use CREATE INDEX CONCURRENTLY for non-blocking creation
                 sql_create_meta_index = SQL(
                     "CREATE INDEX CONCURRENTLY {index_name} ON {schema_name}.{table_name} USING GIN ({column_name}) WITH (fastupdate = off)"
                 ).format(
                     index_name=index_name_identifier,
                     schema_name=Identifier(self.schema_name),
                     table_name=Identifier(self.table_name),
                     column_name=Identifier("meta"), # Index the 'meta' column
                 )
                 metadata_index_queries[index_name_identifier] = sql_create_meta_index
             else:
                logger.warning("'metadata_to_index' is provided, but 'metadata_schema' is not. Cannot create indices for specific keys. Only 'meta' can be indexed.")


        return (
            sql_table_exists,
            sql_create_table,
            sql_index_exists, # Generic query to check index existence
            keyword_index_name_identifier, # Name of the keyword index
            sql_create_keyword_index, # SQL to create keyword index
            metadata_index_queries, # Dictionary of {index_name: create_sql} for metadata
        )

    def _initialize_table(self):
        """
        Internal method to initialize the table.
        """
        if self.recreate_table:
            self.delete_table()

        (
            sql_table_exists,
            sql_create_table,
            sql_index_exists,
            keyword_index_name,
            sql_create_keyword_index,
            metadata_index_queries,
        ) = self._build_table_creation_queries()

        table_exists = bool(
            self._execute_sql(
                sql_table_exists, (self.schema_name, self.table_name), "Could not check if table exists"
            ).fetchone()
        )
        if not table_exists:
            self._execute_sql(sql_create_table, error_msg="Could not create table")

        index_exists = bool(
            self._execute_sql(
                sql_index_exists, # Use generic index check query
                (self.schema_name, self.table_name, str(keyword_index_name)), # Check for specific index name
                "Could not check if keyword index exists",
            ).fetchone()
        )
        if not index_exists:
            self._execute_sql(sql_create_keyword_index, error_msg="Could not create keyword index on table")

        # Create metadata indices if specified
        for index_name, sql_create_meta_index in metadata_index_queries.items():
            meta_index_exists = bool(
                self._execute_sql(
                    sql_index_exists, # Use generic index check query
                    (self.schema_name, self.table_name, str(index_name)), # Check for specific index name
                    f"Could not check if metadata index '{str(index_name)}' exists",
                ).fetchone()
            )
            if not meta_index_exists:
                self._execute_sql(sql_create_meta_index, error_msg=f"Could not create metadata index '{str(index_name)}' on table")
            else:
                logger.info(f"Metadata index '{str(index_name)}' already exists. Skipping creation.")

        if self.search_strategy == "hnsw":
            self._handle_hnsw()

        self._table_initialized = True

    async def _initialize_table_async(self):
        """
        Internal async method to initialize the table.
        """
        if self.recreate_table:
            await self.delete_table_async()

        (
            sql_table_exists,
            sql_create_table,
            sql_index_exists,
            keyword_index_name,
            sql_create_keyword_index,
            metadata_index_queries,
        ) = self._build_table_creation_queries()

        table_exists = bool(
            await (
                await self._execute_sql_async(
                    sql_table_exists,
                    (self.schema_name, self.table_name),
                    "Could not check if table exists",
                    self._async_cursor,
                )
            ).fetchone()
        )
        if not table_exists:
            await self._execute_sql_async(sql_create_table, error_msg="Could not create table")

        index_exists = bool(
            await (
                await self._execute_sql_async(
                    sql_index_exists, # Use generic index check query
                    (self.schema_name, self.table_name, str(keyword_index_name)), # Check for specific index name
                    "Could not check if keyword index exists",
                     self._async_cursor,
                )
            ).fetchone()
        )
        if not index_exists:
            await self._execute_sql_async(sql_create_keyword_index, error_msg="Could not create keyword index on table")

        # Create metadata indices if specified (async)
        for index_name, sql_create_meta_index in metadata_index_queries.items():
            meta_index_exists = bool(
                 await (
                    await self._execute_sql_async(
                        sql_index_exists, # Use generic index check query
                        (self.schema_name, self.table_name, str(index_name)), # Check for specific index name
                        f"Could not check if metadata index '{str(index_name)}' exists",
                         self._async_cursor,
                    )
                ).fetchone()
            )
            if not meta_index_exists:
                 await self._execute_sql_async(sql_create_meta_index, error_msg=f"Could not create metadata index '{str(index_name)}' on table")
            else:
                logger.info(f"Metadata index '{str(index_name)}' already exists. Skipping creation.")


        if self.search_strategy == "hnsw":
            await self._handle_hnsw_async()

        self._table_initialized = True

    def delete_table(self):
        """
        Deletes the table used to store Haystack documents.
        The name of the schema (`schema_name`) and the name of the table (`table_name`)
        are defined when initializing the `PgvectorDocumentStore`.
        """
        delete_sql = SQL("DROP TABLE IF EXISTS {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        self._execute_sql(
            delete_sql,
            error_msg=f"Could not delete table {self.schema_name}.{self.table_name} in PgvectorDocumentStore",
        )

    async def delete_table_async(self):
        """
        Async method to delete the table used to store Haystack documents.
        """
        delete_sql = SQL("DROP TABLE IF EXISTS {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        await self._execute_sql_async(
            delete_sql,
            error_msg=f"Could not delete table {self.schema_name}.{self.table_name} in PgvectorDocumentStore",
        )

    def _build_hnsw_queries(self):
        """Common method to build all HNSW-related SQL queries"""

        sql_set_hnsw_ef_search = (
            SQL("SET hnsw.ef_search = {hnsw_ef_search}").format(hnsw_ef_search=SQLLiteral(self.hnsw_ef_search))
            if self.hnsw_ef_search
            else None
        )

        sql_hnsw_index_exists = SQL(
            "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND tablename = %s AND indexname = %s"
        )

        sql_drop_hnsw_index = SQL("DROP INDEX IF EXISTS {schema_name}.{index_name}").format(
            schema_name=Identifier(self.schema_name),
            index_name=Identifier(self.hnsw_index_name),
        )

        pg_ops = VECTOR_FUNCTION_TO_POSTGRESQL_OPS[self.vector_function]

        # Use CREATE INDEX CONCURRENTLY for non-blocking creation
        sql_create_hnsw_index = SQL(
            "CREATE INDEX CONCURRENTLY {index_name} ON {schema_name}.{table_name} USING hnsw (embedding {ops})"
        ).format(
            schema_name=Identifier(self.schema_name),
            index_name=Identifier(self.hnsw_index_name),
            table_name=Identifier(self.table_name),
            ops=SQL(pg_ops),
        )

        # Add creation kwargs if any valid ones exist
        valid_kwargs = {
            k: v for k, v in self.hnsw_index_creation_kwargs.items() if k in HNSW_INDEX_CREATION_VALID_KWARGS
        }
        if valid_kwargs:
            kwargs_str = ", ".join(f"{k} = {v}" for k, v in valid_kwargs.items())
            sql_create_hnsw_index += SQL(" WITH ({})").format(SQL(kwargs_str))

        return sql_set_hnsw_ef_search, sql_hnsw_index_exists, sql_drop_hnsw_index, sql_create_hnsw_index

    def _handle_hnsw(self):
        """
        Internal method to handle the HNSW index creation.
        It also sets the `hnsw.ef_search` parameter for queries if it is specified.
        """

        sql_set_hnsw_ef_search, sql_hnsw_index_exists, sql_drop_hnsw_index, sql_create_hnsw_index = (
            self._build_hnsw_queries()
        )

        if self.hnsw_ef_search:
            self._execute_sql(sql_set_hnsw_ef_search, error_msg="Could not set hnsw.ef_search")

        index_exists = bool(
            self._execute_sql(
                sql_hnsw_index_exists,
                (self.schema_name, self.table_name, self.hnsw_index_name),
                "Could not check if HNSW index exists",
            ).fetchone()
        )

        if index_exists and not self.hnsw_recreate_index_if_exists:
            logger.warning(
                "HNSW index already exists and won't be recreated. "
                "If you want to recreate it, pass 'hnsw_recreate_index_if_exists=True' to the "
                "Document Store constructor"
            )
            return

        self._execute_sql(sql_drop_hnsw_index, error_msg="Could not drop HNSW index")

        self._execute_sql(sql_create_hnsw_index, error_msg="Could not create HNSW index")

    async def _handle_hnsw_async(self):
        """
        Internal async method to handle the HNSW index creation.
        """

        sql_set_hnsw_ef_search, sql_hnsw_index_exists, sql_drop_hnsw_index, sql_create_hnsw_index = (
            self._build_hnsw_queries()
        )

        if self.hnsw_ef_search:
            await self._execute_sql_async(sql_set_hnsw_ef_search, error_msg="Could not set hnsw.ef_search")

        index_exists = bool(
            await (
                await self._execute_sql_async(
                    sql_hnsw_index_exists,
                    (self.schema_name, self.table_name, self.hnsw_index_name),
                    "Could not check if HNSW index exists",
                    self._async_cursor,
                )
            ).fetchone()
        )

        if index_exists and not self.hnsw_recreate_index_if_exists:
            logger.warning(
                "HNSW index already exists and won't be recreated. "
                "If you want to recreate it, pass 'hnsw_recreate_index_if_exists=True' to the "
                "Document Store constructor"
            )
            return

        await self._execute_sql_async(sql_drop_hnsw_index, error_msg="Could not drop HNSW index")

        await self._execute_sql_async(sql_create_hnsw_index, error_msg="Could not create HNSW index")

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns:
            Number of documents in the document store.
        """
        sql_count = SQL("SELECT COUNT(*) FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        self._ensure_db_setup()
        count = self._execute_sql(sql_count, error_msg="Could not count documents in PgvectorDocumentStore").fetchone()[
            0
        ]
        return count

    async def count_documents_async(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns:
            Number of documents in the document store.
        """
        sql_count = SQL("SELECT COUNT(*) FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        await self._ensure_db_setup_async()

        result = await (
            await self._execute_sql_async(sql_count, error_msg="Could not count documents in PgvectorDocumentStore")
        ).fetchone()

        return result[0]

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :raises TypeError: If `filters` is not a dictionary.
        :raises ValueError: If `filters` syntax is invalid.
        :returns: A list of Documents that match the given filters.
        """
        _validate_filters(filters)

        sql_filter = SQL("SELECT * FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(
                filters=filters,
                original_schema=self.original_metadata_schema,
                sanitized_schema=self.sanitized_metadata_schema
            )
            sql_filter += sql_where_clause

        self._ensure_db_setup()
        result = self._execute_sql(
            sql_filter,
            params,
            error_msg="Could not filter documents from PgvectorDocumentStore.",
            cursor=self._dict_cursor,
        )

        records = result.fetchall()
        docs = _from_pg_to_haystack_documents(records)
        return docs

    async def filter_documents_async(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Asynchronously returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.

        :raises TypeError: If `filters` is not a dictionary.
        :raises ValueError: If `filters` syntax is invalid.
        :returns: A list of Documents that match the given filters.
        """
        _validate_filters(filters)

        sql_filter = SQL("SELECT * FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
        )

        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(
                filters=filters,
                original_schema=self.original_metadata_schema,
                sanitized_schema=self.sanitized_metadata_schema
            )
            sql_filter += sql_where_clause

        await self._ensure_db_setup_async()
        result = await self._execute_sql_async(
            sql_filter,
            params,
            error_msg="Could not filter documents from PgvectorDocumentStore.",
            cursor=self._async_dict_cursor,
        )

        records = await result.fetchall()
        docs = _from_pg_to_haystack_documents(records)
        return docs

    def _build_insert_statement(self, policy: DuplicatePolicy) -> sql.Composed:
        """
        Builds the SQL insert statement to write documents, potentially including
        dynamic columns based on metadata_schema.
        NOTE: This does NOT include the generated 'content_fts' column, as it's
              populated automatically by PostgreSQL.
        """
        # Base columns that are always present (excluding generated columns like content_fts)
        columns = [
            Identifier("id"),
            Identifier("embedding"),
            Identifier("content"),
            Identifier("blob_data"),
            Identifier("blob_meta"),
            Identifier("blob_mime_type"),
        ]
        # Corresponding placeholders
        placeholders = [
            SQL("%(id)s"),
            SQL("%(embedding)s"),
            SQL("%(content)s"),
            SQL("%(blob_data)s"),
            SQL("%(blob_meta)s"),
            SQL("%(blob_mime_type)s"),
        ]

        # Add metadata columns if schema is defined
        if self.sanitized_metadata_schema:
            sanitized_meta_keys = [Identifier(key) for key in self.sanitized_metadata_schema.keys()]
            columns.extend(sanitized_meta_keys)
            placeholders.extend([SQL(f"%({key})s") for key in self.sanitized_metadata_schema.keys()])
            logger.debug(f"Building INSERT for flat metadata columns: {sanitized_meta_keys}")
        else:
            # Otherwise, add the single 'meta' JSONB column
            columns.append(Identifier("meta"))
            placeholders.append(SQL("%(meta)s"))
            logger.debug("Building INSERT for single 'meta' JSONB column")


        # Build the main INSERT INTO ... VALUES ... part
        sql_insert = SQL("INSERT INTO {schema_name}.{table_name} ({columns}) VALUES ({placeholders})").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            columns=SQL(", ").join(columns),
            placeholders=SQL(", ").join(placeholders),
        )

        # Build the ON CONFLICT part based on the policy
        if policy == DuplicatePolicy.OVERWRITE:
            # Build "SET col1 = EXCLUDED.col1, col2 = EXCLUDED.col2, ..."
            # Exclude 'id' column from update set
            # Also EXCLUDE the 'content_fts' column as it's generated
            update_columns = []
            for col in columns: # Iterate through the columns we are *inserting*
                if col.string != 'id': # Don't update id on conflict
                    update_columns.append(col)

            update_assignments = []
            for col in update_columns:
                 # col is already an Identifier
                 # Make sure we update based on EXCLUDED values for inserted columns
                update_assignments.append(SQL("{col} = EXCLUDED.{col}").format(col=col))
            # IMPORTANT: The generated column 'content_fts' will be updated automatically by PG
            # when its dependent column ('content') is updated via EXCLUDED.content.
            # We don't need to (and shouldn't) include it in the SET clause.

            sql_conflict = SQL(" ON CONFLICT (id) DO UPDATE SET ") + SQL(", ").join(update_assignments)
            sql_insert += sql_conflict
            logger.debug("Using ON CONFLICT DO UPDATE policy")
        elif policy == DuplicatePolicy.SKIP:
            sql_insert += SQL(" ON CONFLICT (id) DO NOTHING")
            logger.debug("Using ON CONFLICT DO NOTHING policy")
        elif policy == DuplicatePolicy.FAIL:
             # Default behavior of INSERT is to fail on conflict, so no extra clause needed
             logger.debug("Using ON CONFLICT FAIL policy (default INSERT behavior)")
             pass # No additional clause needed


        sql_insert += SQL(" RETURNING id")

        # Ensure the final query is a Composed object for executemany
        if isinstance(sql_insert, str):
             # This case shouldn't typically happen with psycopg3 sql module usage, but as a safeguard
             return sql.SQL(sql_insert)
        elif isinstance(sql_insert, sql.Composed):
             return sql_insert
        else:
             # Wrap other SQL objects if necessary
             return sql.Composed([sql_insert])


    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents to the document store.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises ValueError: If `documents` contains objects that are not of type `Document`.
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
             and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
        :raises DocumentStoreError: If the write operation fails for any other reason.
        :returns: The number of documents written to the document store.
        """
        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        # Pass both original and sanitized metadata_schema to the conversion function
        db_documents = _from_haystack_to_pg_documents(documents, self.original_metadata_schema, self.sanitized_metadata_schema)

        # If no documents result from the conversion (e.g., filtering), return early.
        if not db_documents:
            return 0

        sql_insert = self._build_insert_statement(policy)

        self._ensure_db_setup()
        assert self._cursor is not None  # verified in _ensure_db_setup() but mypy doesn't know that
        assert self._connection is not None  # verified in _ensure_db_setup() but mypy doesn't know that

        sql_query_str = sql_insert.as_string(self._cursor) if not isinstance(sql_insert, str) else sql_insert
        logger.debug("SQL query: {query}\nParameters: {parameters}", query=sql_query_str, parameters=db_documents)

        try:
            self._cursor.executemany(sql_insert, db_documents, returning=True)
        except IntegrityError as ie:
            self._connection.rollback()
            raise DuplicateDocumentError from ie
        except Error as e:
            self._connection.rollback()
            error_msg = (
                "Could not write documents to PgvectorDocumentStore. \n"
                "You can find the SQL query and the parameters in the debug logs."
            )
            raise DocumentStoreError(error_msg) from e

        # get the number of the inserted documents, inspired by psycopg3 docs
        # https://www.psycopg.org/psycopg3/docs/api/cursors.html#psycopg.Cursor.executemany
        written_docs = 0
        # Iterate through the results returned by executemany with returning=True
        # The cursor manages moving between result sets automatically when iterated.
        for _ in self._cursor:
            written_docs += 1

        return written_docs

    async def write_documents_async(
        self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Asynchronously writes documents to the document store.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises ValueError: If `documents` contains objects that are not of type `Document`.
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
             and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
        :raises DocumentStoreError: If the write operation fails for any other reason.
        :returns: The number of documents written to the document store.
        """
        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        # Pass both original and sanitized metadata_schema to the conversion function
        db_documents = _from_haystack_to_pg_documents(documents, self.original_metadata_schema, self.sanitized_metadata_schema)

        # If no documents result from the conversion (e.g., filtering), return early.
        if not db_documents:
            return 0

        sql_insert = self._build_insert_statement(policy)

        await self._ensure_db_setup_async()
        assert self._async_cursor is not None  # verified in _ensure_db_setup_async() but mypy doesn't know that
        assert self._async_connection is not None  # verified in _ensure_db_setup_async() but mypy doesn't know that

        sql_query_str = sql_insert.as_string(self._async_cursor) if not isinstance(sql_insert, str) else sql_insert
        logger.debug("SQL query: {query}\nParameters: {parameters}", query=sql_query_str, parameters=db_documents)

        try:
            await self._async_cursor.executemany(sql_insert, db_documents, returning=True)
        except IntegrityError as ie:
            await self._async_connection.rollback()
            raise DuplicateDocumentError from ie
        except Error as e:
            await self._async_connection.rollback()
            error_msg = (
                "Could not write documents to PgvectorDocumentStore. \n"
                "You can find the SQL query and the parameters in the debug logs."
            )
            raise DocumentStoreError(error_msg) from e

        written_docs = 0
        async for _ in self._async_cursor:
            written_docs += 1

        return written_docs

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        if not document_ids:
            return

        document_ids_str = ", ".join(f"'{document_id}'" for document_id in document_ids)

        delete_sql = SQL("DELETE FROM {schema_name}.{table_name} WHERE id IN ({document_ids_str})").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            document_ids_str=SQL(document_ids_str),
        )

        self._ensure_db_setup()
        self._execute_sql(delete_sql, error_msg="Could not delete documents from PgvectorDocumentStore")

    async def delete_documents_async(self, document_ids: List[str]) -> None:
        """
        Asynchronously deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        if not document_ids:
            return

        document_ids_str = ", ".join(f"'{document_id}'" for document_id in document_ids)

        delete_sql = SQL("DELETE FROM {schema_name}.{table_name} WHERE id IN ({document_ids_str})").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            document_ids_str=SQL(document_ids_str),
        )

        await self._ensure_db_setup_async()
        await self._execute_sql_async(delete_sql, error_msg="Could not delete documents from PgvectorDocumentStore")

    def _build_keyword_retrieval_query(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None):
        """
        Builds the SQL query and the where parameters for keyword retrieval using
        the pre-generated 'content_fts' tsvector column.
        """
        sql_select = SQL(
            """
            SELECT {table_name}.*, ts_rank_cd(content_fts, query) AS score
            FROM {schema_name}.{table_name}, plainto_tsquery({language}, %s) query
            WHERE content_fts @@ query
            """
        ).format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            language=SQLLiteral(self.language),
        )

        where_params = ()
        sql_where_clause = SQL("")
        if filters:
            # Filter conversion needs to happen *after* the FTS WHERE clause.
            # We add it using AND.
            filter_where_clause, where_params = _convert_filters_to_where_clause_and_params(
                filters=filters,
                operator="AND", # Use AND to combine with the FTS condition
                original_schema=self.original_metadata_schema,
                sanitized_schema=self.sanitized_metadata_schema
            )
            # Prepend AND to the filter clause if it's not empty
            if filter_where_clause.string.strip():
                 sql_where_clause = SQL(" AND ") + filter_where_clause


        sql_sort = SQL(" ORDER BY score DESC LIMIT {top_k}").format(top_k=SQLLiteral(top_k))

        # Combine: SELECT ... FROM ... WHERE FTS @@ query [AND filters] ORDER BY ... LIMIT ...
        sql_query = sql_select + sql_where_clause + sql_sort

        # The parameters now consist of the user query string FIRST (for plainto_tsquery),
        # followed by any parameters generated by the filter conversion.
        return sql_query, where_params # Params for filters will be appended later when executing

    def _keyword_retrieval(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Retrieves documents that match the query using a full-text search on the
        pre-generated 'content_fts' column.

        This method is not meant to be part of the public interface of
        `PgvectorDocumentStore` and it should not be called directly.
        `PgvectorKeywordRetriever` uses this method directly and is the public interface for it.

        :returns: List of Documents that match the `query`
        """
        if not query:
            msg = "query must be a non-empty string"
            raise ValueError(msg)

        _validate_filters(filters) # Validate filters early

        sql_query, filter_params = self._build_keyword_retrieval_query(query=query, top_k=top_k, filters=filters)

        self._ensure_db_setup()
        # Execute with the query string first, then filter parameters
        all_params = (query,) + filter_params
        result = self._execute_sql(
            sql_query,
            all_params,
            error_msg="Could not retrieve documents using keyword search from PgvectorDocumentStore.",
            cursor=self._dict_cursor,
        )

        records = result.fetchall()
        docs = _from_pg_to_haystack_documents(records)
        return docs

    async def _keyword_retrieval_async(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Asynchronously retrieves documents that match the query using a full-text search
        on the pre-generated 'content_fts' column.
        """
        if not query:
            msg = "query must be a non-empty string"
            raise ValueError(msg)

        _validate_filters(filters) # Validate filters early

        sql_query, filter_params = self._build_keyword_retrieval_query(query=query, top_k=top_k, filters=filters)

        await self._ensure_db_setup_async()
        # Execute with the query string first, then filter parameters
        all_params = (query,) + filter_params
        result = await self._execute_sql_async(
            sql_query,
            all_params,
            error_msg="Could not retrieve documents using keyword search from PgvectorDocumentStore.",
            cursor=self._async_dict_cursor,
        )

        records = await result.fetchall()
        docs = _from_pg_to_haystack_documents(records)
        return docs

    def _check_and_build_embedding_retrieval_query(
        self,
        query_embedding: List[float],
        vector_function: Optional[Literal["cosine_similarity", "inner_product", "l2_distance"]],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """
        Performs checks and builds the SQL query and the where parameters for embedding retrieval.
        """

        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)
        if len(query_embedding) != self.embedding_dimension:
            msg = (
                f"query_embedding dimension ({len(query_embedding)}) does not match PgvectorDocumentStore "
                f"embedding dimension ({self.embedding_dimension})."
            )
            raise ValueError(msg)

        vector_function = vector_function or self.vector_function
        if vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}, but got {vector_function}"
            raise ValueError(msg)

        # the vector must be a string with this format: "'[3,1,2]'"
        query_embedding_for_postgres = f"'[{','.join(str(el) for el in query_embedding)}]'"

        # to compute the scores, we use the approach described in pgvector README:
        # https://github.com/pgvector/pgvector?tab=readme-ov-file#distances
        # cosine_similarity and inner_product are modified from the result of the operator
        if vector_function == "cosine_similarity":
            score_definition = f"1 - (embedding <=> {query_embedding_for_postgres}) AS score"
        elif vector_function == "inner_product":
            score_definition = f"(embedding <#> {query_embedding_for_postgres}) * -1 AS score"
        elif vector_function == "l2_distance":
            score_definition = f"embedding <-> {query_embedding_for_postgres} AS score"

        sql_select = SQL("SELECT *, {score} FROM {schema_name}.{table_name}").format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            score=SQL(score_definition),
        )

        sql_where_clause = SQL("")
        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(
                filters=filters,
                original_schema=self.original_metadata_schema,
                sanitized_schema=self.sanitized_metadata_schema
            )

        # we always want to return the most similar documents first
        # so when using l2_distance, the sort order must be ASC
        sort_order = "ASC" if vector_function == "l2_distance" else "DESC"

        sql_sort = SQL(" ORDER BY score {sort_order} LIMIT {top_k}").format(
            top_k=SQLLiteral(top_k),
            sort_order=SQL(sort_order),
        )

        sql_query = sql_select + sql_where_clause + sql_sort

        return sql_query, params

    def _embedding_retrieval(
        self,
        query_embedding: List[float],
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        vector_function: Optional[Literal["cosine_similarity", "inner_product", "l2_distance"]] = None,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.

        This method is not meant to be part of the public interface of
        `PgvectorDocumentStore` and it should not be called directly.
        `PgvectorEmbeddingRetriever` uses this method directly and is the public interface for it.

        :returns: List of Documents that are most similar to `query_embedding`
        """

        sql_query, params = self._check_and_build_embedding_retrieval_query(
            query_embedding=query_embedding, vector_function=vector_function, top_k=top_k, filters=filters
        )
        self._ensure_db_setup()
        result = self._execute_sql(
            sql_query,
            params,
            error_msg="Could not retrieve documents from PgvectorDocumentStore.",
            cursor=self._dict_cursor,
        )

        records = result.fetchall()
        docs = _from_pg_to_haystack_documents(records)
        return docs

    async def _embedding_retrieval_async(
        self,
        query_embedding: List[float],
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        vector_function: Optional[Literal["cosine_similarity", "inner_product", "l2_distance"]] = None,
    ) -> List[Document]:
        """
        Asynchronously retrieves documents that are most similar to the query embedding using a
        vector similarity metric.
        """

        sql_query, params = self._check_and_build_embedding_retrieval_query(
            query_embedding=query_embedding, vector_function=vector_function, top_k=top_k, filters=filters
        )

        await self._ensure_db_setup_async()
        result = await self._execute_sql_async(
            sql_query,
            params,
            error_msg="Could not retrieve documents from PgvectorDocumentStore.",
            cursor=self._async_dict_cursor,
        )

        records = await result.fetchall()
        docs = _from_pg_to_haystack_documents(records)
        return docs

    # Helper method to sanitize metadata keys into valid SQL column names
    def _sanitize_metadata_keys(self, keys: List[str]) -> List[Identifier]:
        sanitized_keys = []
        for key in keys:
            # Replace non-alphanumeric characters with underscores
            # Ensure it starts with a letter or underscore
            sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in key)
            if not sanitized or not (sanitized[0].isalpha() or sanitized[0] == '_'):
                sanitized = '_' + sanitized
            # Truncate to PostgreSQL's default max identifier length (63) if necessary
            sanitized_keys.append(Identifier(sanitized[:63]))
        return sanitized_keys

    # Helper method to sanitize the entire metadata schema dictionary
    def _sanitize_metadata_schema(self, schema: Dict[str, str]) -> Dict[str, str]:
        sanitized_schema = {}
        for key, value in schema.items():
            # Replace non-alphanumeric characters with underscores
            # Ensure it starts with a letter or underscore
            sanitized_key = ''.join(c if c.isalnum() or c == '_' else '_' for c in key)
            if not sanitized_key or not (sanitized_key[0].isalpha() or sanitized_key[0] == '_'):
                sanitized_key = '_' + sanitized_key
            # Truncate to PostgreSQL's default max identifier length (63) if necessary
            # We store the sanitized key string here, not the Identifier yet
            sanitized_schema[sanitized_key[:63]] = value
        return sanitized_schema
