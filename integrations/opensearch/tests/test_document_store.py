# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import random
from typing import List
from unittest.mock import patch

import pytest
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests
from haystack.utils.auth import Secret
from opensearchpy.exceptions import RequestError

from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from haystack_integrations.document_stores.opensearch.auth import AWSAuth
from haystack_integrations.document_stores.opensearch.document_store import DEFAULT_MAX_CHUNK_BYTES


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some hosts")
    res = document_store.to_dict()
    assert res == {
        "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
        "init_parameters": {
            "embedding_dim": 768,
            "hosts": "some hosts",
            "index": "default",
            "mappings": {
                "dynamic_templates": [{"strings": {"mapping": {"type": "keyword"}, "match_mapping_type": "string"}}],
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {"dimension": 768, "index": True, "type": "knn_vector"},
                },
            },
            "max_chunk_bytes": DEFAULT_MAX_CHUNK_BYTES,
            "method": None,
            "settings": {"index.knn": True},
            "return_embedding": False,
            "create_index": True,
            "http_auth": None,
            "use_ssl": None,
            "verify_certs": None,
            "timeout": None,
        },
    }


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict(_mock_opensearch_client):
    data = {
        "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "index": "default",
            "max_chunk_bytes": 1000,
            "embedding_dim": 1536,
            "create_index": False,
            "return_embedding": True,
            "aws_service": "es",
            "http_auth": ("admin", "admin"),
            "use_ssl": True,
            "verify_certs": True,
            "timeout": 60,
        },
    }
    document_store = OpenSearchDocumentStore.from_dict(data)
    assert document_store._hosts == "some hosts"
    assert document_store._index == "default"
    assert document_store._max_chunk_bytes == 1000
    assert document_store._embedding_dim == 1536
    assert document_store._method is None
    assert document_store._mappings == {
        "properties": {
            "embedding": {"type": "knn_vector", "index": True, "dimension": 1536},
            "content": {"type": "text"},
        },
        "dynamic_templates": [
            {
                "strings": {
                    "match_mapping_type": "string",
                    "mapping": {"type": "keyword"},
                }
            }
        ],
    }
    assert document_store._settings == {"index.knn": True}
    assert document_store._return_embedding is True
    assert document_store._create_index is False
    assert document_store._http_auth == ("admin", "admin")
    assert document_store._use_ssl is True
    assert document_store._verify_certs is True
    assert document_store._timeout == 60


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_init_is_lazy(_mock_opensearch_client):
    OpenSearchDocumentStore(hosts="testhost")
    _mock_opensearch_client.assert_not_called()


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_get_default_mappings(_mock_opensearch_client):
    store = OpenSearchDocumentStore(hosts="testhost", embedding_dim=1536, method={"name": "hnsw"})
    assert store._mappings["properties"]["embedding"] == {
        "type": "knn_vector",
        "index": True,
        "dimension": 1536,
        "method": {"name": "hnsw"},
    }


class TestAuth:
    @pytest.fixture(autouse=True)
    def mock_boto3_session(self):
        with patch("boto3.Session") as mock_client:
            yield mock_client

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_init_with_basic_auth(self, _mock_opensearch_client):
        document_store = OpenSearchDocumentStore(hosts="testhost", http_auth=("user", "pw"))
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert _mock_opensearch_client.call_args[1]["http_auth"] == ("user", "pw")

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_init_without_auth(self, _mock_opensearch_client):
        document_store = OpenSearchDocumentStore(hosts="testhost")
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert _mock_opensearch_client.call_args[1]["http_auth"] is None

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_init_aws_auth(self, _mock_opensearch_client):
        document_store = OpenSearchDocumentStore(
            hosts="testhost",
            http_auth=AWSAuth(aws_region_name=Secret.from_token("dummy-region")),
            use_ssl=True,
            verify_certs=True,
        )
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert isinstance(_mock_opensearch_client.call_args[1]["http_auth"], AWSAuth)
        assert _mock_opensearch_client.call_args[1]["use_ssl"] is True
        assert _mock_opensearch_client.call_args[1]["verify_certs"] is True

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_from_dict_basic_auth(self, _mock_opensearch_client):
        document_store = OpenSearchDocumentStore.from_dict(
            {
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
                "init_parameters": {
                    "hosts": "testhost",
                    "http_auth": ["user", "pw"],
                    "use_ssl": True,
                    "verify_certs": True,
                },
            }
        )
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert _mock_opensearch_client.call_args[1]["http_auth"] == ["user", "pw"]

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_from_dict_aws_auth(self, _mock_opensearch_client, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "dummy-region")
        document_store = OpenSearchDocumentStore.from_dict(
            {
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
                "init_parameters": {
                    "hosts": "testhost",
                    "http_auth": {
                        "type": "haystack_integrations.document_stores.opensearch.auth.AWSAuth",
                        "init_parameters": {},
                    },
                    "use_ssl": True,
                    "verify_certs": True,
                },
            }
        )
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert isinstance(_mock_opensearch_client.call_args[1]["http_auth"], AWSAuth)
        assert _mock_opensearch_client.call_args[1]["use_ssl"] is True
        assert _mock_opensearch_client.call_args[1]["verify_certs"] is True

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_to_dict_basic_auth(self, _mock_opensearch_client):
        document_store = OpenSearchDocumentStore(hosts="some hosts", http_auth=("user", "pw"))
        res = document_store.to_dict()
        assert res == {
            "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            "init_parameters": {
                "embedding_dim": 768,
                "hosts": "some hosts",
                "index": "default",
                "mappings": {
                    "dynamic_templates": [
                        {"strings": {"mapping": {"type": "keyword"}, "match_mapping_type": "string"}}
                    ],
                    "properties": {
                        "content": {"type": "text"},
                        "embedding": {"dimension": 768, "index": True, "type": "knn_vector"},
                    },
                },
                "max_chunk_bytes": DEFAULT_MAX_CHUNK_BYTES,
                "method": None,
                "settings": {"index.knn": True},
                "return_embedding": False,
                "create_index": True,
                "http_auth": ("user", "pw"),
                "use_ssl": None,
                "verify_certs": None,
                "timeout": None,
            },
        }

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_to_dict_aws_auth(self, _mock_opensearch_client, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "dummy-region")
        document_store = OpenSearchDocumentStore(hosts="some hosts", http_auth=AWSAuth())
        res = document_store.to_dict()
        assert res == {
            "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            "init_parameters": {
                "embedding_dim": 768,
                "hosts": "some hosts",
                "index": "default",
                "mappings": {
                    "dynamic_templates": [
                        {"strings": {"mapping": {"type": "keyword"}, "match_mapping_type": "string"}}
                    ],
                    "properties": {
                        "content": {"type": "text"},
                        "embedding": {"dimension": 768, "index": True, "type": "knn_vector"},
                    },
                },
                "max_chunk_bytes": DEFAULT_MAX_CHUNK_BYTES,
                "method": None,
                "settings": {"index.knn": True},
                "return_embedding": False,
                "create_index": True,
                "http_auth": {
                    "type": "haystack_integrations.document_stores.opensearch.auth.AWSAuth",
                    "init_parameters": {
                        "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                        "aws_secret_access_key": {
                            "type": "env_var",
                            "env_vars": ["AWS_SECRET_ACCESS_KEY"],
                            "strict": False,
                        },
                        "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                        "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                        "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                        "aws_service": "es",
                    },
                },
                "use_ssl": None,
                "verify_certs": None,
                "timeout": None,
            },
        }

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_init_with_env_var_secrets(self, _mock_opensearch_client, monkeypatch):
        """Test the default initialization using environment variables"""
        monkeypatch.setenv("OPENSEARCH_USERNAME", "user")
        monkeypatch.setenv("OPENSEARCH_PASSWORD", "pass")

        document_store = OpenSearchDocumentStore(hosts="testhost")
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert _mock_opensearch_client.call_args[1]["http_auth"] == ["user", "pass"]

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_init_with_missing_env_vars(self, _mock_opensearch_client):
        """Test that auth is None when environment variables are missing"""
        document_store = OpenSearchDocumentStore(hosts="testhost")
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert _mock_opensearch_client.call_args[1]["http_auth"] is None

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_to_dict_with_env_var_secrets(self, _mock_opensearch_client, monkeypatch):
        """Test serialization with environment variables"""
        monkeypatch.setenv("OPENSEARCH_USERNAME", "user")
        monkeypatch.setenv("OPENSEARCH_PASSWORD", "pass")

        document_store = OpenSearchDocumentStore(hosts="testhost")
        serialized = document_store.to_dict()

        assert "http_auth" in serialized["init_parameters"]
        auth = serialized["init_parameters"]["http_auth"]
        assert isinstance(auth, list)
        assert len(auth) == 2
        # Check that we have two Secret dictionaries with correct env vars
        assert auth[0]["type"] == "env_var"
        assert auth[0]["env_vars"] == ["OPENSEARCH_USERNAME"]
        assert auth[1]["type"] == "env_var"
        assert auth[1]["env_vars"] == ["OPENSEARCH_PASSWORD"]

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    def test_from_dict_with_env_var_secrets(self, _mock_opensearch_client, monkeypatch):
        """Test deserialization with environment variables"""
        # Set environment variables so the secrets resolve properly
        monkeypatch.setenv("OPENSEARCH_USERNAME", "user")
        monkeypatch.setenv("OPENSEARCH_PASSWORD", "pass")

        data = {
            "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
            "init_parameters": {
                "hosts": "testhost",
                "http_auth": [
                    {"type": "env_var", "env_vars": ["OPENSEARCH_USERNAME"], "strict": False},
                    {"type": "env_var", "env_vars": ["OPENSEARCH_PASSWORD"], "strict": False},
                ],
            },
        }
        document_store = OpenSearchDocumentStore.from_dict(data)
        document_store._ensure_initialized()
        assert document_store._client
        _mock_opensearch_client.assert_called_once()
        assert _mock_opensearch_client.call_args[1]["http_auth"] == ["user", "pass"]


@pytest.mark.integration
class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def document_store(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["https://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=768,
            method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
        )
        yield store
        store._ensure_initialized()
        assert store._client
        store._client.indices.delete(index=index, params={"ignore": [400, 404]})

    @pytest.fixture
    def document_store_readonly(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["https://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=768,
            method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
            create_index=False,
        )
        store._ensure_initialized()
        assert store._client
        store._client.cluster.put_settings(body={"transient": {"action.auto_create_index": False}})
        yield store
        store._client.cluster.put_settings(body={"transient": {"action.auto_create_index": True}})
        store._client.indices.delete(index=index, params={"ignore": [400, 404]})

    @pytest.fixture
    def document_store_embedding_dim_4(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["https://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=4,
            method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
        )
        yield store
        store._client.indices.delete(index=index, params={"ignore": [400, 404]})

    @pytest.fixture
    def document_store_embedding_dim_4_faiss(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["https://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=4,
            method={"space_type": "innerproduct", "engine": "faiss", "name": "hnsw"},
        )
        yield store
        store._client.indices.delete(index=index, params={"ignore": [400, 404]})

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        """
        The OpenSearchDocumentStore.filter_documents() method returns a Documents with their score set.
        We don't want to compare the score, so we set it to None before comparing the documents.
        """
        received_meta = []
        for doc in received:
            r = {
                "number": doc.meta.get("number"),
                "name": doc.meta.get("name"),
            }
            received_meta.append(r)

        expected_meta = []
        for doc in expected:
            r = {
                "number": doc.meta.get("number"),
                "name": doc.meta.get("name"),
            }
            expected_meta.append(r)
        for doc in received:
            doc.score = None

        super().assert_documents_are_equal(received, expected)

    def test_write_documents(self, document_store: OpenSearchDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_write_documents_readonly(self, document_store_readonly: OpenSearchDocumentStore):
        docs = [Document(id="1")]
        with pytest.raises(DocumentStoreError, match="index_not_found_exception"):
            document_store_readonly.write_documents(docs)

    def test_create_index(self, document_store_readonly: OpenSearchDocumentStore):
        document_store_readonly.create_index()
        assert document_store_readonly._client.indices.exists(index=document_store_readonly._index)

    def test_bm25_retrieval(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
            ]
        )

        res = document_store._bm25_retrieval("functional", top_k=3)
        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    def test_bm25_retrieval_pagination(self, document_store: OpenSearchDocumentStore):
        """
        Test that handling of pagination works as expected, when the matching documents are > 10.
        """
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
                Document(content="Java is an object oriented programming language"),
                Document(content="Javascript is a programming language"),
                Document(content="Typescript is a programming language"),
                Document(content="C is a programming language"),
            ]
        )

        res = document_store._bm25_retrieval("programming", top_k=11)
        assert len(res) == 11
        assert all("programming" in doc.content for doc in res)

    def test_bm25_retrieval_all_terms_must_match(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
            ]
        )

        res = document_store._bm25_retrieval("functional Haskell", top_k=3, all_terms_must_match=True)
        assert len(res) == 1
        assert "Haskell is a functional programming language" in res[0].content

    def test_bm25_retrieval_all_terms_must_match_false(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
            ]
        )

        res = document_store._bm25_retrieval("functional Haskell", top_k=10, all_terms_must_match=False)
        assert len(res) == 5
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content
        assert "functional" in res[3].content
        assert "functional" in res[4].content

    def test_bm25_retrieval_with_fuzziness(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
            ]
        )

        query_with_typo = "functinal"
        # Query without fuzziness to search for the exact match
        res = document_store._bm25_retrieval(query_with_typo, top_k=3, fuzziness="0")
        # Nothing is found as the query contains a typo
        assert res == []

        # Query with fuzziness with the same query
        res = document_store._bm25_retrieval(query_with_typo, top_k=3, fuzziness="1")
        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    def test_bm25_retrieval_with_filters(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(
                    content="Haskell is a functional programming language",
                    meta={"likes": 100000, "language_type": "functional"},
                    id="1",
                ),
                Document(
                    content="Lisp is a functional programming language",
                    meta={"likes": 10000, "language_type": "functional"},
                    id="2",
                ),
                Document(
                    content="Exilir is a functional programming language",
                    meta={"likes": 1000, "language_type": "functional"},
                    id="3",
                ),
                Document(
                    content="F# is a functional programming language",
                    meta={"likes": 100, "language_type": "functional"},
                    id="4",
                ),
                Document(
                    content="C# is a functional programming language",
                    meta={"likes": 10, "language_type": "functional"},
                    id="5",
                ),
                Document(
                    content="C++ is an object oriented programming language",
                    meta={"likes": 100000, "language_type": "object_oriented"},
                    id="6",
                ),
                Document(
                    content="Dart is an object oriented programming language",
                    meta={"likes": 10000, "language_type": "object_oriented"},
                    id="7",
                ),
                Document(
                    content="Go is an object oriented programming language",
                    meta={"likes": 1000, "language_type": "object_oriented"},
                    id="8",
                ),
                Document(
                    content="Python is a object oriented programming language",
                    meta={"likes": 100, "language_type": "object_oriented"},
                    id="9",
                ),
                Document(
                    content="Ruby is a object oriented programming language",
                    meta={"likes": 10, "language_type": "object_oriented"},
                    id="10",
                ),
                Document(
                    content="PHP is a object oriented programming language",
                    meta={"likes": 1, "language_type": "object_oriented"},
                    id="11",
                ),
            ]
        )

        res = document_store._bm25_retrieval(
            "programming",
            top_k=10,
            filters={"field": "language_type", "operator": "==", "value": "functional"},
        )
        assert len(res) == 5
        retrieved_ids = sorted([doc.id for doc in res])
        assert retrieved_ids == ["1", "2", "3", "4", "5"]

    def test_bm25_retrieval_with_custom_query(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(
                    content="Haskell is a functional programming language",
                    meta={"likes": 100000, "language_type": "functional"},
                    id="1",
                ),
                Document(
                    content="Lisp is a functional programming language",
                    meta={"likes": 10000, "language_type": "functional"},
                    id="2",
                ),
                Document(
                    content="Exilir is a functional programming language",
                    meta={"likes": 1000, "language_type": "functional"},
                    id="3",
                ),
                Document(
                    content="F# is a functional programming language",
                    meta={"likes": 100, "language_type": "functional"},
                    id="4",
                ),
                Document(
                    content="C# is a functional programming language",
                    meta={"likes": 10, "language_type": "functional"},
                    id="5",
                ),
                Document(
                    content="C++ is an object oriented programming language",
                    meta={"likes": 100000, "language_type": "object_oriented"},
                    id="6",
                ),
                Document(
                    content="Dart is an object oriented programming language",
                    meta={"likes": 10000, "language_type": "object_oriented"},
                    id="7",
                ),
                Document(
                    content="Go is an object oriented programming language",
                    meta={"likes": 1000, "language_type": "object_oriented"},
                    id="8",
                ),
                Document(
                    content="Python is a object oriented programming language",
                    meta={"likes": 100, "language_type": "object_oriented"},
                    id="9",
                ),
                Document(
                    content="Ruby is a object oriented programming language",
                    meta={"likes": 10, "language_type": "object_oriented"},
                    id="10",
                ),
                Document(
                    content="PHP is a object oriented programming language",
                    meta={"likes": 1, "language_type": "object_oriented"},
                    id="11",
                ),
            ]
        )

        custom_query = {
            "query": {
                "function_score": {
                    "query": {"bool": {"must": {"match": {"content": "$query"}}, "filter": "$filters"}},
                    "field_value_factor": {"field": "likes", "factor": 0.1, "modifier": "log1p", "missing": 0},
                }
            }
        }

        res = document_store._bm25_retrieval(
            "functional",
            top_k=3,
            custom_query=custom_query,
            filters={"field": "language_type", "operator": "==", "value": "functional"},
        )
        assert len(res) == 3
        assert "1" == res[0].id
        assert "2" == res[1].id
        assert "3" == res[2].id

    def test_bm25_retrieval_with_custom_query_empty_filters(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(
                    content="Haskell is a functional programming language",
                    meta={"likes": 100000, "language_type": "functional"},
                    id="1",
                ),
                Document(
                    content="Lisp is a functional programming language",
                    meta={"likes": 10000, "language_type": "functional"},
                    id="2",
                ),
                Document(
                    content="Exilir is a functional programming language",
                    meta={"likes": 1000, "language_type": "functional"},
                    id="3",
                ),
                Document(
                    content="F# is a functional programming language",
                    meta={"likes": 100, "language_type": "functional"},
                    id="4",
                ),
                Document(
                    content="C# is a functional programming language",
                    meta={"likes": 10, "language_type": "functional"},
                    id="5",
                ),
                Document(
                    content="C++ is an object oriented programming language",
                    meta={"likes": 100000, "language_type": "object_oriented"},
                    id="6",
                ),
                Document(
                    content="Dart is an object oriented programming language",
                    meta={"likes": 10000, "language_type": "object_oriented"},
                    id="7",
                ),
                Document(
                    content="Go is an object oriented programming language",
                    meta={"likes": 1000, "language_type": "object_oriented"},
                    id="8",
                ),
                Document(
                    content="Python is a object oriented programming language",
                    meta={"likes": 100, "language_type": "object_oriented"},
                    id="9",
                ),
                Document(
                    content="Ruby is a object oriented programming language",
                    meta={"likes": 10, "language_type": "object_oriented"},
                    id="10",
                ),
                Document(
                    content="PHP is a object oriented programming language",
                    meta={"likes": 1, "language_type": "object_oriented"},
                    id="11",
                ),
            ]
        )

        custom_query = {
            "query": {
                "function_score": {
                    "query": {"bool": {"must": {"match": {"content": "$query"}}, "filter": "$filters"}},
                    "field_value_factor": {"field": "likes", "factor": 0.1, "modifier": "log1p", "missing": 0},
                }
            }
        }

        res = document_store._bm25_retrieval(
            "functional",
            top_k=3,
            custom_query=custom_query,
        )
        assert len(res) == 3
        assert "1" == res[0].id
        assert "2" == res[1].id
        assert "3" == res[2].id

    def test_embedding_retrieval(self, document_store_embedding_dim_4: OpenSearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_embedding_dim_4.write_documents(docs)
        results = document_store_embedding_dim_4._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters={}
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document"
        assert results[1].content == "2nd best document"

    def test_embedding_retrieval_with_filters(self, document_store_embedding_dim_4: OpenSearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(
                content="Not very similar document with meta field",
                embedding=[0.0, 0.8, 0.3, 0.9],
                meta={"meta_field": "custom_value"},
            ),
        ]
        document_store_embedding_dim_4.write_documents(docs)

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}
        # we set top_k=3, to make the test pass as we are not sure whether efficient filtering is supported for nmslib
        # TODO: remove top_k=3, when efficient filtering is supported for nmslib
        results = document_store_embedding_dim_4._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=3, filters=filters
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    def test_embedding_retrieval_with_filters_efficient_filtering(
        self, document_store_embedding_dim_4_faiss: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(
                content="Not very similar document with meta field",
                embedding=[0.0, 0.8, 0.3, 0.9],
                meta={"meta_field": "custom_value"},
            ),
        ]
        document_store_embedding_dim_4_faiss.write_documents(docs)

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}
        results = document_store_embedding_dim_4_faiss._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1],
            filters=filters,
            efficient_filtering=True,
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    def test_embedding_retrieval_pagination(self, document_store_embedding_dim_4: OpenSearchDocumentStore):
        """
        Test that handling of pagination works as expected, when the matching documents are > 10.
        """

        docs = [
            Document(content=f"Document {i}", embedding=[random.random() for _ in range(4)])  # noqa: S311
            for i in range(20)
        ]

        document_store_embedding_dim_4.write_documents(docs)
        results = document_store_embedding_dim_4._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=11, filters={}
        )
        assert len(results) == 11

    def test_embedding_retrieval_with_custom_query(self, document_store_embedding_dim_4: OpenSearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(
                content="Not very similar document with meta field",
                embedding=[0.0, 0.8, 0.3, 0.9],
                meta={"meta_field": "custom_value"},
            ),
        ]
        document_store_embedding_dim_4.write_documents(docs)

        custom_query = {
            "query": {
                "bool": {"must": [{"knn": {"embedding": {"vector": "$query_embedding", "k": 3}}}], "filter": "$filters"}
            }
        }

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}
        results = document_store_embedding_dim_4._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=1, filters=filters, custom_query=custom_query
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    def test_embedding_retrieval_query_documents_different_embedding_sizes(
        self, document_store_embedding_dim_4: OpenSearchDocumentStore
    ):
        """
        Test that the retrieval fails if the query embedding and the documents have different embedding sizes.
        """
        docs = [Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4])]
        document_store_embedding_dim_4.write_documents(docs)

        with pytest.raises(RequestError):
            document_store_embedding_dim_4._embedding_retrieval(query_embedding=[0.1, 0.1])

    def test_write_documents_different_embedding_sizes_fail(
        self, document_store_embedding_dim_4: OpenSearchDocumentStore
    ):
        """
        Test that write_documents fails if the documents have different embedding sizes.
        """
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Hello world", embedding=[0.1, 0.2]),
        ]

        with pytest.raises(DocumentStoreError):
            document_store_embedding_dim_4.write_documents(docs)

    @patch("haystack_integrations.document_stores.opensearch.document_store.bulk")
    def test_write_documents_with_badly_formatted_bulk_errors(self, mock_bulk, document_store):
        error = {"some_key": "some_value"}
        mock_bulk.return_value = ([], [error])

        with pytest.raises(DocumentStoreError) as e:
            document_store.write_documents([Document(content="Hello world")])
            e.match(f"{error}")

    @patch("haystack_integrations.document_stores.opensearch.document_store.bulk")
    def test_write_documents_max_chunk_bytes(self, mock_bulk, document_store):
        mock_bulk.return_value = (1, [])
        document_store.write_documents([Document(content="Hello world")])

        assert mock_bulk.call_args.kwargs["max_chunk_bytes"] == DEFAULT_MAX_CHUNK_BYTES

    @pytest.fixture
    def document_store_no_embbding_returned(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["https://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=4,
            return_embedding=False,
            method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
        )
        yield store
        store._client.indices.delete(index=index, params={"ignore": [400, 404]})

    def test_embedding_retrieval_but_dont_return_embeddings_for_embedding_retrieval(
        self, document_store_no_embbding_returned: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_no_embbding_returned.write_documents(docs)
        results = document_store_no_embbding_returned._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters={}
        )
        assert len(results) == 2
        assert results[0].embedding is None

    def test_embedding_retrieval_but_dont_return_embeddings_for_bm25_retrieval(
        self, document_store_no_embbding_returned: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_no_embbding_returned.write_documents(docs)
        results = document_store_no_embbding_returned._bm25_retrieval("document", top_k=2)
        assert len(results) == 2
        assert results[0].embedding is None


@pytest.mark.integration
class TestDocumentStoreAsync:

    @pytest.fixture
    async def document_store(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["https://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=768,
            method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
        )
        yield store
        store._ensure_initialized()
        assert store._client
        store._client.indices.delete(index=index, params={"ignore": [400, 404]})
        await store._async_client.close()

    @pytest.fixture
    async def document_store_readonly(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["https://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=768,
            method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
            create_index=False,
        )
        store._ensure_initialized()
        assert store._client
        store._client.cluster.put_settings(body={"transient": {"action.auto_create_index": False}})
        yield store
        store._client.cluster.put_settings(body={"transient": {"action.auto_create_index": True}})
        store._client.indices.delete(index=index, params={"ignore": [400, 404]})
        await store._async_client.close()

    @pytest.fixture
    async def document_store_embedding_dim_4(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["https://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=4,
            method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
        )
        yield store
        store._ensure_initialized()
        assert store._client
        store._client.indices.delete(index=index, params={"ignore": [400, 404]})
        await store._async_client.close()

    @pytest.fixture
    async def document_store_no_embbding_returned(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["https://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=4,
            return_embedding=False,
            method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
        )
        store._ensure_initialized()
        yield store
        store._client.indices.delete(index=index, params={"ignore": [400, 404]})

    @pytest.mark.asyncio
    async def test_write_documents(self, document_store: OpenSearchDocumentStore):
        assert await document_store.write_documents_async([Document(id="1")]) == 1

    @pytest.mark.asyncio
    async def test_bm25_retrieval(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
            ]
        )
        res = await document_store._bm25_retrieval_async("functional", top_k=3)
        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    @pytest.mark.asyncio
    async def test_bm25_retrieval_pagination(self, document_store: OpenSearchDocumentStore):
        """
        Test that handling of pagination works as expected, when the matching documents are > 10.
        """
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
                Document(content="Java is an object oriented programming language"),
                Document(content="Javascript is a programming language"),
                Document(content="Typescript is a programming language"),
                Document(content="C is a programming language"),
            ]
        )

        res = await document_store._bm25_retrieval_async("programming", top_k=11)
        assert len(res) == 11
        assert all("programming" in doc.content for doc in res)

    @pytest.mark.asyncio
    async def test_bm25_retrieval_all_terms_must_match(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
            ]
        )

        res = await document_store._bm25_retrieval_async("functional Haskell", top_k=3, all_terms_must_match=True)
        assert len(res) == 1
        assert "Haskell is a functional programming language" in res[0].content

    @pytest.mark.asyncio
    async def test_bm25_retrieval_all_terms_must_match_false(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
            ]
        )

        res = await document_store._bm25_retrieval_async("functional Haskell", top_k=10, all_terms_must_match=False)
        assert len(res) == 5
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content
        assert "functional" in res[3].content
        assert "functional" in res[4].content

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_filters(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(
                    content="Haskell is a functional programming language",
                    meta={"likes": 100000, "language_type": "functional"},
                    id="1",
                ),
                Document(
                    content="Lisp is a functional programming language",
                    meta={"likes": 10000, "language_type": "functional"},
                    id="2",
                ),
                Document(
                    content="Exilir is a functional programming language",
                    meta={"likes": 1000, "language_type": "functional"},
                    id="3",
                ),
                Document(
                    content="F# is a functional programming language",
                    meta={"likes": 100, "language_type": "functional"},
                    id="4",
                ),
                Document(
                    content="C# is a functional programming language",
                    meta={"likes": 10, "language_type": "functional"},
                    id="5",
                ),
                Document(
                    content="C++ is an object oriented programming language",
                    meta={"likes": 100000, "language_type": "object_oriented"},
                    id="6",
                ),
                Document(
                    content="Dart is an object oriented programming language",
                    meta={"likes": 10000, "language_type": "object_oriented"},
                    id="7",
                ),
                Document(
                    content="Go is an object oriented programming language",
                    meta={"likes": 1000, "language_type": "object_oriented"},
                    id="8",
                ),
                Document(
                    content="Python is a object oriented programming language",
                    meta={"likes": 100, "language_type": "object_oriented"},
                    id="9",
                ),
                Document(
                    content="Ruby is a object oriented programming language",
                    meta={"likes": 10, "language_type": "object_oriented"},
                    id="10",
                ),
                Document(
                    content="PHP is a object oriented programming language",
                    meta={"likes": 1, "language_type": "object_oriented"},
                    id="11",
                ),
            ]
        )
        res = await document_store._bm25_retrieval_async(
            "programming",
            top_k=10,
            filters={"field": "language_type", "operator": "==", "value": "functional"},
        )
        assert len(res) == 5
        retrieved_ids = sorted([doc.id for doc in res])
        assert retrieved_ids == ["1", "2", "3", "4", "5"]

    @pytest.mark.asyncio
    async def test_bm25_retrieval_with_custom_query(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(
                    content="Haskell is a functional programming language",
                    meta={"likes": 100000, "language_type": "functional"},
                    id="1",
                ),
                Document(
                    content="Lisp is a functional programming language",
                    meta={"likes": 10000, "language_type": "functional"},
                    id="2",
                ),
                Document(
                    content="Exilir is a functional programming language",
                    meta={"likes": 1000, "language_type": "functional"},
                    id="3",
                ),
                Document(
                    content="F# is a functional programming language",
                    meta={"likes": 100, "language_type": "functional"},
                    id="4",
                ),
                Document(
                    content="C# is a functional programming language",
                    meta={"likes": 10, "language_type": "functional"},
                    id="5",
                ),
                Document(
                    content="C++ is an object oriented programming language",
                    meta={"likes": 100000, "language_type": "object_oriented"},
                    id="6",
                ),
                Document(
                    content="Dart is an object oriented programming language",
                    meta={"likes": 10000, "language_type": "object_oriented"},
                    id="7",
                ),
                Document(
                    content="Go is an object oriented programming language",
                    meta={"likes": 1000, "language_type": "object_oriented"},
                    id="8",
                ),
                Document(
                    content="Python is a object oriented programming language",
                    meta={"likes": 100, "language_type": "object_oriented"},
                    id="9",
                ),
                Document(
                    content="Ruby is a object oriented programming language",
                    meta={"likes": 10, "language_type": "object_oriented"},
                    id="10",
                ),
                Document(
                    content="PHP is a object oriented programming language",
                    meta={"likes": 1, "language_type": "object_oriented"},
                    id="11",
                ),
            ]
        )

        custom_query = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must": {"match": {"content": "$query"}},
                            "filter": "$filters",
                        }
                    },
                    "field_value_factor": {
                        "field": "likes",
                        "factor": 0.1,
                        "modifier": "log1p",
                        "missing": 0,
                    },
                }
            }
        }
        res = await document_store._bm25_retrieval_async(
            "functional",
            top_k=3,
            custom_query=custom_query,
            filters={"field": "language_type", "operator": "==", "value": "functional"},
        )
        assert len(res) == 3
        assert "1" == res[0].id
        assert "2" == res[1].id
        assert "3" == res[2].id

    @pytest.mark.asyncio
    async def test_embedding_retrieval(self, document_store_embedding_dim_4: OpenSearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_embedding_dim_4.write_documents(docs)

        results = await document_store_embedding_dim_4._embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters={}
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document"
        assert results[1].content == "2nd best document"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_with_filters(self, document_store_embedding_dim_4: OpenSearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(
                content="Not very similar document with meta field",
                embedding=[0.0, 0.8, 0.3, 0.9],
                meta={"meta_field": "custom_value"},
            ),
        ]
        document_store_embedding_dim_4.write_documents(docs)

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}

        results = await document_store_embedding_dim_4._embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=3, filters=filters
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_with_custom_query(self, document_store_embedding_dim_4: OpenSearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(
                content="Not very similar document with meta field",
                embedding=[0.0, 0.8, 0.3, 0.9],
                meta={"meta_field": "custom_value"},
            ),
        ]
        document_store_embedding_dim_4.write_documents(docs)

        custom_query = {
            "query": {
                "bool": {
                    "must": [{"knn": {"embedding": {"vector": "$query_embedding", "k": 3}}}],
                    "filter": "$filters",
                }
            }
        }

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}

        results = await document_store_embedding_dim_4._embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1],
            top_k=1,
            filters=filters,
            custom_query=custom_query,
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    @pytest.mark.asyncio
    async def test_embedding_retrieval_but_dont_return_embeddings_for_embedding_retrieval(
        self, document_store_no_embbding_returned: OpenSearchDocumentStore
    ):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_no_embbding_returned.write_documents(docs)

        results = await document_store_no_embbding_returned._embedding_retrieval_async(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters={}
        )
        assert len(results) == 2
        assert results[0].embedding is None

    @pytest.mark.asyncio
    async def test_count_documents(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
            [
                Document(content="test doc 1"),
                Document(content="test doc 2"),
                Document(content="test doc 3"),
            ]
        )
        assert await document_store.count_documents_async() == 3

    @pytest.mark.asyncio
    async def test_filter_documents(self, document_store: OpenSearchDocumentStore):
        filterable_docs = [
            Document(
                content="1",
                meta={
                    "number": -10,
                },
            ),
            Document(
                content="2",
                meta={
                    "number": 100,
                },
            ),
        ]
        await document_store.write_documents_async(filterable_docs)
        result = await document_store.filter_documents_async(
            filters={"field": "meta.number", "operator": "==", "value": 100}
        )
        TestDocumentStore().assert_documents_are_equal(
            result, [d for d in filterable_docs if d.meta.get("number") == 100]
        )

    @pytest.mark.asyncio
    async def test_delete_documents(self, document_store: OpenSearchDocumentStore):
        doc = Document(content="test doc")
        await document_store.write_documents_async([doc])
        assert document_store.count_documents() == 1

        await document_store.delete_documents_async([doc.id])
        assert await document_store.count_documents_async() == 0
