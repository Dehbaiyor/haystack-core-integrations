# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, Literal, Optional, Tuple

from haystack.errors import FilterError
from psycopg.sql import SQL, Identifier
from psycopg.types.json import Jsonb
import logging

logger = logging.getLogger(__name__)

# we need this mapping to cast meta values to the correct type,
# since they are stored in the JSONB field as strings.
# this dict can be extended if needed
PYTHON_TYPES_TO_PG_TYPES = {
    int: "integer",
    float: "real",
    bool: "boolean",
}

NO_VALUE = "no_value"


def _validate_filters(filters: Optional[Dict[str, Any]] = None):
    """
    Validates the filters provided.
    """
    if filters:
        if not isinstance(filters, dict):
            msg = "Filters must be a dictionary"
            raise TypeError(msg)
        if "operator" not in filters and "conditions" not in filters:
            msg = "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
            raise ValueError(msg)


def _convert_filters_to_where_clause_and_params(
    filters: Dict[str, Any],
    operator: Literal["WHERE", "AND"] = "WHERE",
    original_schema: Optional[Dict[str, str]] = None,
    sanitized_schema: Optional[Dict[str, str]] = None
) -> Tuple[SQL, Tuple]:
    """
    Convert Haystack filters to a WHERE clause and a tuple of params to query PostgreSQL.
    Takes into account metadata schema for potentially querying flat columns.
    """
    if "field" in filters:
        query, values = _parse_comparison_condition(filters, original_schema, sanitized_schema)
    else:
        query, values = _parse_logical_condition(filters, original_schema, sanitized_schema)

    sql_query = SQL(query) if isinstance(query, str) else query
    where_clause = SQL(f" {operator} ") + sql_query
    params = tuple(value for value in values if value != NO_VALUE)

    return where_clause, params


def _parse_logical_condition(
    condition: Dict[str, Any],
    original_schema: Optional[Dict[str, str]] = None,
    sanitized_schema: Optional[Dict[str, str]] = None
) -> Tuple[SQL, List[Any]]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    if operator not in ["AND", "OR"]:
        msg = f"Unknown logical operator '{operator}'. Valid operators are: 'AND', 'OR'"
        raise FilterError(msg)

    # logical conditions can be nested, so we need to parse them recursively
    conditions = []
    for c in condition["conditions"]:
        if "field" in c:
            query, vals = _parse_comparison_condition(c, original_schema, sanitized_schema)
        else:
            query, vals = _parse_logical_condition(c, original_schema, sanitized_schema)
        conditions.append((query, vals))

    query_parts, values = [], []
    for c in conditions:
        query_parts.append(c[0])
        values.append(c[1])
    if isinstance(values[0], list):
        values = list(chain.from_iterable(values))

    if operator == "AND":
        sql_query = SQL(" AND ").join([SQL(q) if isinstance(q, str) else q for q in query_parts])
    elif operator == "OR":
        sql_query = SQL(" OR ").join([SQL(q) if isinstance(q, str) else q for q in query_parts])
    else:
        msg = f"Unknown logical operator '{operator}'"
        raise FilterError(msg)

    return SQL("({})").format(sql_query), values


def _parse_comparison_condition(
    condition: Dict[str, Any],
    original_schema: Optional[Dict[str, str]] = None,
    sanitized_schema: Optional[Dict[str, str]] = None
) -> Tuple[SQL, List[Any]]:
    field_name: str = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'. Valid operators are: {list(COMPARISON_OPERATORS.keys())}"
        raise FilterError(msg)

    value: Any = condition["value"]

    # Check if it's a metadata field and if we have a schema
    if field_name.startswith("meta.") and sanitized_schema and original_schema:
        original_field_key = field_name.split(".", 1)[-1]

        # Check if this original key exists in the user-provided schema
        if original_field_key in original_schema:
            # Key exists in schema, query the flat column
            # Find the sanitized key (re-sanitizing for now)
            # TODO: Refactor sanitization logic into a shared utility.
            temp_sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in original_field_key)
            if not temp_sanitized or not (temp_sanitized[0].isalpha() or temp_sanitized[0] == '_'):
                temp_sanitized = '_' + temp_sanitized
            sanitized_key = temp_sanitized[:63]

            # Ensure the derived sanitized key is actually in the sanitized schema map
            if sanitized_key in sanitized_schema:
                # Use the sanitized key as the SQL identifier for the flat column
                field_identifier = Identifier(sanitized_key)
                # Get the SQL snippet and value from the operator function
                sql_template_str, value_list = COMPARISON_OPERATORS[operator](field_identifier, value)
                # Format the template using the identifier
                # Note: Operator functions must return a format string expecting {field} or handle Identifier directly.
                # Assuming they return a string like "{field} = %s" or similar.
                # Let's adjust the operator functions slightly if needed, or handle it here.
                # For now, assume COMPARISON_OPERATORS functions return (sql_string_template, value_list_or_item)
                # We format the field identifier into the template here.
                if isinstance(sql_template_str, str) and "{field}" in sql_template_str:
                     # Simple string formatting might be unsafe if template isn't controlled.
                     # Better: Assume operator funcs return Tuple[Union[SQL, str], Any]
                     # If str, wrap in SQL here.
                     sql_part = SQL(sql_template_str).format(field=field_identifier)
                elif isinstance(sql_template_str, SQL):
                     # If the function already returns SQL, assume it handled the Identifier
                     sql_part = sql_template_str
                else:
                     # Fallback or error? Let's assume string template for now.
                     # This might need refinement based on COMPARISON_OPERATORS implementation.
                     sql_part = SQL(sql_template_str).format(field=field_identifier)

                # Ensure value is always a list
                if not isinstance(value_list, list):
                    value_list = [value_list]

                return sql_part, value_list
            else:
                 # Should not happen if schemas are consistent, but log a warning
                 logger.warning(f"Sanitized key '{sanitized_key}' derived from '{original_field_key}' not found in sanitized schema. Falling back to JSONB.")

        # If original key not in original_schema, fall through to treat as JSONB

    # --- Fallback or Non-meta field logic --- 
    field_sql_str : str
    if field_name.startswith("meta."):
        # Use existing JSONB logic if schema doesn't apply or field not in schema
        field_sql_str = _treat_meta_field(field_name, value)
    else:
        # Handle non-meta fields directly (e.g., "content", "id")
        # Quote if not already quoted? Use Identifier for safety.
        # Assuming simple field names are safe for now, but Identifier is better.
        field_sql_str = str(Identifier(field_name))

    # Get the SQL template and value from the operator function
    sql_template_str, value_list = COMPARISON_OPERATORS[operator](field_sql_str, value)

    # Ensure value is always a list
    if not isinstance(value_list, list):
        value_list = [value_list]

    # Safely format the field string into the SQL template
    # Note: field_sql_str might contain casts like '(meta->>...)'. SQL() handles quoting.
    sql_part = SQL(sql_template_str).format(field=SQL(field_sql_str))

    return sql_part, value_list


def _treat_meta_field(field: str, value: Any) -> str:
    """
    Internal method that modifies the field str
    to make the meta JSONB field queryable.

    Examples:
    >>> _treat_meta_field(field="meta.number", value=9)
    "(meta->>'number')::integer"

    >>> _treat_meta_field(field="meta.name", value="my_name")
    "meta->>'name'"
    """

    # use the ->> operator to access keys in the meta JSONB field
    field_name = field.split(".", 1)[-1]
    field = f"meta->>'{field_name}'"

    # meta fields are stored as strings in the JSONB field,
    # so we need to cast them to the correct type
    type_value = PYTHON_TYPES_TO_PG_TYPES.get(type(value))
    if isinstance(value, list) and len(value) > 0:
        type_value = PYTHON_TYPES_TO_PG_TYPES.get(type(value[0]))

    if type_value:
        field = f"({field})::{type_value}"

    return field


def _equal(field: Any, value: Any) -> Tuple[str, Any]:
    field_sql = SQL("{field}").format(field=field) if isinstance(field, Identifier) else SQL(field)
    if value is None:
        return SQL("{field_sql} IS NULL").format(field_sql=field_sql), NO_VALUE
    return SQL("{field_sql} = %s").format(field_sql=field_sql), value


def _not_equal(field: Any, value: Any) -> Tuple[str, Any]:
    field_sql = SQL("{field}").format(field=field) if isinstance(field, Identifier) else SQL(field)
    return SQL("{field_sql} IS DISTINCT FROM %s").format(field_sql=field_sql), value


def _greater_than(field: Any, value: Any) -> Tuple[str, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, Jsonb]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    field_sql = SQL("{field}").format(field=field) if isinstance(field, Identifier) else SQL(field)
    return SQL("{field_sql} > %s").format(field_sql=field_sql), value


def _greater_than_equal(field: Any, value: Any) -> Tuple[str, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, Jsonb]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    field_sql = SQL("{field}").format(field=field) if isinstance(field, Identifier) else SQL(field)
    return SQL("{field_sql} >= %s").format(field_sql=field_sql), value


def _less_than(field: Any, value: Any) -> Tuple[str, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, Jsonb]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    field_sql = SQL("{field}").format(field=field) if isinstance(field, Identifier) else SQL(field)
    return SQL("{field_sql} < %s").format(field_sql=field_sql), value


def _less_than_equal(field: Any, value: Any) -> Tuple[str, Any]:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, Jsonb]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)
    field_sql = SQL("{field}").format(field=field) if isinstance(field, Identifier) else SQL(field)
    return SQL("{field_sql} <= %s").format(field_sql=field_sql), value


def _not_in(field: Any, value: Any) -> Tuple[str, List]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'not in' comparator in Pinecone"
        raise FilterError(msg)

    return SQL("{field} IS NULL OR {field} != ALL(%s)").format(field=field), [value]


def _in(field: Any, value: Any) -> Tuple[str, List]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' comparator in Pinecone"
        raise FilterError(msg)

    # see https://www.psycopg.org/psycopg3/docs/basic/adapt.html#lists-adaptation
    return SQL("{field} = ANY(%s)").format(field=field), [value]


def _like(field: Any, value: Any) -> Tuple[str, Any]:
    if not isinstance(value, str):
        msg = f"{field}'s value must be a str when using 'LIKE' "
        raise FilterError(msg)
    field_sql = SQL("{field}").format(field=field) if isinstance(field, Identifier) else SQL(field)
    return SQL("{field_sql} LIKE %s").format(field_sql=field_sql), value


def _not_like(field: Any, value: Any) -> Tuple[str, Any]:
    if not isinstance(value, str):
        msg = f"{field}'s value must be a str when using 'LIKE' "
        raise FilterError(msg)
    field_sql = SQL("{field}").format(field=field) if isinstance(field, Identifier) else SQL(field)
    return SQL("{field_sql} NOT LIKE %s").format(field_sql=field_sql), value


COMPARISON_OPERATORS = {
    "==": _equal,
    "!=": _not_equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
    "in": _in,
    "not in": _not_in,
    "like": _like,
    "not like": _not_like,
}
