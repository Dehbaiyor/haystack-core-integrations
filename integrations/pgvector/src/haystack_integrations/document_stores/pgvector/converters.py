from typing import Any, Dict, List, Optional, Tuple

from haystack import logging
from haystack.dataclasses import ByteStream, Document
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)


def _from_haystack_to_pg_documents(
    documents: List[Document],
    original_schema: Optional[Dict[str, str]] = None,
    sanitized_schema: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Internal method to convert a list of Haystack Documents to a list of dictionaries that can be used to insert
    documents into the PgvectorDocumentStore.

    If sanitized_schema is provided, it populates flat metadata columns based on the schema.
    Otherwise, it populates the 'meta' JSONB field.
    """

    db_documents = []
    for document in documents:
        # Start with basic document fields, excluding score, blob, and meta (handled separately)
        db_document = {k: v for k, v in document.to_dict(flatten=False).items() if k not in ["score", "blob", "meta"]}

        # Handle blob data
        blob = document.blob
        db_document["blob_data"] = blob.data if blob else None
        db_document["blob_meta"] = Jsonb(blob.meta) if blob and blob.meta else None
        db_document["blob_mime_type"] = blob.mime_type if blob and blob.mime_type else None

        original_meta = document.meta or {}

        # Handle metadata based on whether a schema was provided
        if sanitized_schema and original_schema:
            # Populate flat columns using sanitized keys from the schema
            for sanitized_key in sanitized_schema.keys():
                # Find the original key that corresponds to this sanitized key.
                # This assumes the order of keys in original_schema and sanitized_schema match
                # after sanitization, which might be fragile. A safer approach would be
                # to pass a mapping if keys can change order during sanitization.
                # Let's try finding the original key by re-sanitizing.
                original_key_found = None
                for key in original_schema.keys():
                    # Re-sanitize the original key to see if it matches the current sanitized_key
                    # This re-uses the sanitization logic, avoiding passing complex structures.
                    # NOTE: Requires the sanitization logic to be accessible or duplicated here.
                    # For now, assume a helper _sanitize_key function exists/is imported.
                    # We need to import or define _sanitize_key. Let's define a simple version here for now.
                    # TODO: Refactor sanitization logic into a shared utility.
                    temp_sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in key)
                    if not temp_sanitized or not (temp_sanitized[0].isalpha() or temp_sanitized[0] == '_'):
                        temp_sanitized = '_' + temp_sanitized
                    temp_sanitized = temp_sanitized[:63]

                    if temp_sanitized == sanitized_key:
                        original_key_found = key
                        break

                if original_key_found:
                    db_document[sanitized_key] = original_meta.get(original_key_found)
                else:
                    # This case should ideally not happen if schemas are consistent
                    logger.warning(f"Could not find original key for sanitized key '{sanitized_key}'. Skipping.")
                    db_document[sanitized_key] = None # Or handle as an error

        else:
            # Fallback: If no schema, store all meta in the JSONB field
            db_document["meta"] = Jsonb(original_meta)

        if "sparse_embedding" in db_document:
            sparse_embedding = db_document.pop("sparse_embedding", None)
            if sparse_embedding:
                logger.warning(
                    "Document {doc_id} has the `sparse_embedding` field set,"
                    "but storing sparse embeddings in Pgvector is not currently supported."
                    "The `sparse_embedding` field will be ignored.",
                    doc_id=db_document["id"],
                )

        db_documents.append(db_document)

    return db_documents


def _from_pg_to_haystack_documents(documents: List[Dict[str, Any]]) -> List[Document]:
    """
    Internal method to convert a list of dictionaries from pgvector to a list of Haystack Documents.
    """

    haystack_documents = []
    for document in documents:
        haystack_dict = dict(document)
        blob_data = haystack_dict.pop("blob_data")
        blob_meta = haystack_dict.pop("blob_meta")
        blob_mime_type = haystack_dict.pop("blob_mime_type")
        
        # Remove generated content_fts field
        haystack_dict.pop("content_fts")

        # convert the embedding to a list of floats
        if document.get("embedding") is not None:
            haystack_dict["embedding"] = document["embedding"].tolist()

        haystack_document = Document.from_dict(haystack_dict)

        if blob_data:
            blob = ByteStream(data=blob_data, meta=blob_meta, mime_type=blob_mime_type)
            haystack_document.blob = blob

        haystack_documents.append(haystack_document)

    return haystack_documents
