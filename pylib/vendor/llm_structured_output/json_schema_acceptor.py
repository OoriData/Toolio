"""
Acceptors for JSON schema validation or constraning LLM generation to JSON
outputs complying with a JSON schema.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Tuple

from .acceptor import (
    TokenAcceptor,
    StateMachineAcceptor,
    SequenceAcceptor,
    TextAcceptor,
    WaitForAcceptor,
)
from .json_acceptor import (
    WhitespaceAcceptor,
    BooleanAcceptor,
    NullAcceptor,
    StringAcceptor,
    StringConstantAcceptor,
    NumberAcceptor,
    ArrayAcceptor,
    ObjectAcceptor,
    prepare_json_acceptor_tries,
)
from .util.tokentrie import TokenTrie


class SchemaNotImplementedError(Exception):
    """
    Raised when a JSON schema uses a feature that hasn't been implemented here yet
    """


class InvalidSchemaError(Exception):
    """
    Raised when the passed JSON schema is invalid, e.g. a required property is not defined
    """


# pylint: disable-next=invalid-name
def ConstSchemaAcceptor(schema: dict):
    """
    Accept a constant string
    """
    return StringConstantAcceptor(schema["const"])


class EnumSchemaAcceptor(StateMachineAcceptor):
    """
    Accept one of several constant strings
    """

    def __init__(self, schema: dict):
        super().__init__(
            [[(StringConstantAcceptor(value), "$") for value in schema["enum"]]]
        )


class StringSchemaAcceptor(StringAcceptor):
    """
    Accept a JSON string that conforms to a JSON schema
    """

    def __init__(self, schema: dict = None):
        super().__init__()
        self.schema = schema or {}

    def min_length(self):
        """
        Returns the minimum string length according to the schema
        """
        return self.schema.get("minLength", 0)

    def max_length(self):
        """
        Returns the maximum string length according to the schema
        """
        return self.schema.get("maxLength", 10000)  # Arbitrary default

    def validate_value(self, value):
        """
        Validate the string value according to the schema
        """
        if len(value) < self.min_length():
            return False
        if len(value) > self.max_length():
            return False
        if "pattern" in self.schema:
            # This would be better done progressively as the cursor advances
            regex = re.compile(self.schema["pattern"])
            if regex.search(value) is None:
                return False
        if "format" in self.schema:
            raise SchemaNotImplementedError("string.format")
        return True

    class Cursor(StringAcceptor.Cursor):
        """
        Cursor for StringSchemaAcceptor
        """

        def __init__(self, acceptor):
            super().__init__(acceptor)
            self.acceptor = acceptor

        def start_transition(self, transition_acceptor, target_state):
            if self.current_state == 1:
                if target_state == "$":
                    return self.length >= self.acceptor.min_length()
                else:
                    return self.length < self.acceptor.max_length()
            return True

        def complete_transition(self, transition_value, target_state, is_end_state):
            if not super().complete_transition(
                transition_value, target_state, is_end_state
            ):
                return False
            if is_end_state:
                return self.acceptor.validate_value(self.get_value())
            return True


class NumberSchemaAcceptor(NumberAcceptor):
    """
    Accept a JSON number that conforms to a JSON schema
    """

    def __init__(self, schema):
        super().__init__()
        self.schema = schema
        self.is_integer = schema["type"] == "integer"
        self.requires_validation = any(
            constraint in schema
            for constraint in [
                "minimum",
                "exclusiveMinimum",
                "maximum",
                "exclusiveMaximum",
                "multipleOf",
            ]
        )

    def validate_value(self, value):
        """
        Validate the number value according to the schema
        """
        if "minimum" in self.schema and value < self.schema["minimum"]:
            return False
        if (
            "exclusiveMinimum" in self.schema
            and value <= self.schema["exclusiveMinimum"]
        ):
            return False
        if "maximum" in self.schema and value > self.schema["maximum"]:
            return False
        if (
            "exclusiveMaximum" in self.schema
            and value >= self.schema["exclusiveMaximum"]
        ):
            return False
        if "multipleOf" in self.schema:
            divisor = self.schema["multipleOf"]
            if value / divisor != value // divisor:
                return False
        return True

    class Cursor(NumberAcceptor.Cursor):
        """
        Cursor for NumberAcceptor
        """

        def __init__(self, acceptor):
            super().__init__(acceptor)
            self.acceptor = acceptor

        def prune(self, trie):
            """
            The parent class uses a collapsed trie and this means in the token-matching
            phase we are always getting numbers made up of the digit "9", which prevents
            correct validation. If we actually have anything to validate, we disable it.
            """
            if self.acceptor.requires_validation:
                return super(NumberAcceptor.Cursor, self).prune(trie)
            return super().prune(trie)

        def start_transition(self, transition_acceptor, target_state):
            if (
                self.acceptor.is_integer
                and self.current_state == 3
                and target_state == 4
            ):
                return False
            return super().start_transition(transition_acceptor, target_state)

        def complete_transition(self, transition_value, target_state, is_end_state):
            if not super().complete_transition(
                transition_value, target_state, is_end_state
            ):
                return False
            if is_end_state:
                return self.acceptor.validate_value(self.get_value())
            return True


class ArraySchemaAcceptor(ArrayAcceptor):
    """
    TODO
    {
      "type": "array",
      "items": { "type": "number" },
      "uniqueItems": true
    }
    {
      "type": "array",
      "prefixItems": [
        { "type": "number" },
        { "type": "string" },
        { "enum": ["Street", "Avenue", "Boulevard"] },
        { "enum": ["NW", "NE", "SW", "SE"] }
      ],
      "items": { "type": "string" } # Constrain additional items to type string
      "items": false # Do not allow items beyond the prefixItem
      "unevaluatedItems": false # All prefixItems are required
      "unevaluatedItems": { "const": "N/A" } # Default value for prefixItems
    }
    {
      "type": "array",
      "contains": {
        "type": "number" # Contains at least one number
      },
      "minContains": 2, # Must contain at least two numbers
      "maxContains": 3 # Must contain at most three numbers
    }
    """

    def __init__(self, schema, context):
        self.schema = schema
        self.context = context
        super().__init__()

    def get_edges(self, state):
        return {
            0: lambda: [(TextAcceptor("["), 1)],
            1: lambda: [(WhitespaceAcceptor(), 2), (TextAcceptor("]"), "$")],
            2: lambda: [(JsonSchemaAcceptor(self.schema["items"], self.context), 3)],
            3: lambda: [(WhitespaceAcceptor(), 4)],
            4: lambda: [
                (SequenceAcceptor([TextAcceptor(","), WhitespaceAcceptor()]), 2),
                (TextAcceptor("]"), "$"),
            ],
        }[state]()

    def min_items(self) -> int:
        """
        Returns the minimum number of items in the array, according to the schema
        """
        return self.schema.get("minItems", 0)

    def max_items(self) -> int:
        """
        Returns the maximum number of items in the array, according to the schema
        """
        return self.schema.get("maxItems", 2**32)  # Arbitrary default

    class Cursor(ArrayAcceptor.Cursor):
        """
        Cursor for ArrayAcceptor
        """

        def __init__(self, acceptor):
            super().__init__(acceptor)
            self.acceptor = acceptor

        def start_transition(self, transition_acceptor, target_state) -> bool:
            if self.current_state == 4 and target_state == 2:
                return len(self.value) < self.acceptor.max_items()
            if target_state == "$":
                return len(self.value) >= self.acceptor.min_items()
            return True


class ObjectSchemaAcceptor(ObjectAcceptor):
    """
    TODO
    {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "credit_card": { "type": "number" },
        "billing_address": { "type": "string" }
      },
      "required": ["name"],
      "dependentRequired": {
        "credit_card": ["billing_address"]
      }
      "dependentSchemas": { # Alternative to dependentRequired
        "credit_card": {
          "properties": {
            "billing_address": { "type": "string" }
          },
          "required": ["billing_address"]
        }
      }
    }
    {
      "type": "object",
      "properties": {
        "street_address": {
          "type": "string"
        },
        "country": {
          "default": "United States of America",
          "enum": ["United States of America", "Canada"]
        }
      },
      "if": {
        "properties": {
          "country": { "const": "United States of America" }
        }
      },
      "then": {
        "properties": {
          "postal_code": { "pattern": "[0-9]{5}(-[0-9]{4})?" }
        }
      },
      "else": {
        "properties": {
          "postal_code": { "pattern": "[A-Z][0-9][A-Z] [0-9][A-Z][0-9]" }
        }
      }
    }
    """

    # $defs, $ref https://json-schema.org/understanding-json-schema/structuring

    def __init__(self, schema, context):
        self.schema = schema
        self.context = context
        self.properties = schema.get("properties", {})
        # Note that, according to the JSON schema specification, additional properties
        # should be allowed by default. The additionalProperties subschema can be used
        # to limit this. But we default to only allowing the properties defined in the
        # schema, because generally we don't want the LLM to generate at will. An
        # exception to this is when no properties are defined in the schema; in that
        # case we don't use this class but the superclass to allow any JSON object. 
        if schema.get("additionalProperties"):
            raise SchemaNotImplementedError("object.additionalProperties")
        self.required_property_names = schema.get("required", [])
        for required_property_name in self.required_property_names:
            if required_property_name not in self.properties:
                raise InvalidSchemaError(f"Required property '{required_property_name}' not defined")
        
        assert self.properties is not None
        super().__init__()

    def get_edges(self, state):
        if state == 3:
            return [
                (
                    ObjectSchemaAcceptor.PropertyAcceptor(
                        prop_name, prop_schema, self.context
                    ),
                    4,
                )
                for prop_name, prop_schema in self.properties.items()
            ]
        else:
            return super().get_edges(state)

    class Cursor(ObjectAcceptor.Cursor):
        """
        Cursor for ObjectAcceptor
        """

        def __init__(self, acceptor):
            super().__init__(acceptor)
            self.acceptor = acceptor

        def start_transition(self, transition_acceptor, target_state) -> bool:
            if target_state == "$":
                return all(
                    prop_name in self.value
                    for prop_name in self.acceptor.required_property_names
                )
            if self.current_state == 3 and target_state == 4:
                # Is a property already set?
                return transition_acceptor.prop_name not in self.value
            if self.current_state == 5 and target_state == 2:
                # Are all allowed properties already set?
                return len(self.value.keys()) < len(self.acceptor.properties)
            return True

    class PropertyAcceptor(ObjectAcceptor.PropertyAcceptor):
        """
        Acceptor for an object property according to the schema
        """

        def __init__(self, prop_name, prop_schema, context):
            self.prop_name = prop_name
            self.prop_schema = prop_schema
            self.prop_context = {
                "defs": context["defs"],
                "path": f"{context['path']}/{prop_name}",
            }
            super().__init__(
                [
                    StringConstantAcceptor(self.prop_name),
                    WhitespaceAcceptor(),
                    TextAcceptor(":"),
                    WhitespaceAcceptor(),
                    JsonSchemaAcceptor(self.prop_schema, self.prop_context),
                ]
            )

        class Cursor(ObjectAcceptor.PropertyAcceptor.Cursor):
            """
            Cursor for ObjectSchemaAcceptor.PropertyAcceptor
            """

            def __init__(self, acceptor):
                super().__init__(acceptor)
                self.acceptor = acceptor

            def complete_transition(
                self, transition_value, target_state, is_end_state
            ) -> bool:
                if not super().complete_transition(
                    transition_value, target_state, is_end_state
                ):
                    return False
                hooks = self.acceptor.prop_schema.get("__hooks", None)
                if hooks:
                    prop_name = self.acceptor.prop_name
                    if target_state == 4 and "value_start" in hooks:
                        hooks["value_start"](prop_name)
                    elif is_end_state and "value_end" in hooks:
                        hooks["value_end"](prop_name, transition_value)
                return True

            def get_value(self):
                return (self.acceptor.prop_name, self.prop_value)


class AnyOfAcceptor(StateMachineAcceptor):
    """
    Accepts JSON input that complies with any of several provided JSON schemas
    """

    def __init__(self, schemas: list[dict], context):
        super().__init__(
            [[(JsonSchemaAcceptor(schema, context), "$") for schema in schemas]]
        )


def merged(dict1, dict2):
    """
    Returns a new dictionary resulting from the merge of two input dictionaries
    """
    copy = dict1.copy()
    copy.update(dict2)
    return copy


class DefinitionNotFoundError(Exception):
    """
    Raised when a JSON schema reference cannot be resolved.
    """

    def __init__(self, ref: str):
        super().__init__(f"Definition not found for schema reference '{ref}'")


def resolve_subschemas(schema, defs, visited_refs) -> list:
    """
    Resolve JSON schema references and schema combination keywords (allOf, anyOf, oneOf)
    """
    if "$ref" in schema:
        schema_ref = schema["$ref"]
        if schema_ref in visited_refs:
            return visited_refs[schema_ref]
        resolved_schema = []
        visited_refs[schema_ref] = resolved_schema
        schema_def = defs.get(schema_ref)
        if schema_def is None:
            raise DefinitionNotFoundError(schema_ref)
        resolved_schema += resolve_subschemas(schema_def, defs, visited_refs)
        return resolved_schema

    if "allOf" in schema:
        merged_schema = schema.copy()
        del merged_schema["allOf"]
        # Resolve any other keywords ("anyOf", "oneOf", ...) before adding subschemas.
        schemas = resolve_subschemas(merged_schema, defs, visited_refs)
        for subschema in schema["allOf"]:
            schemas = [
                merged(ms, rs)
                for ms in schemas
                for rs in resolve_subschemas(subschema, defs, visited_refs)
            ]
        return schemas

    if "anyOf" in schema:
        merged_schema = schema.copy()
        del merged_schema["anyOf"]
        # Resolve any other keywords ("allOf", "oneOf", ...) before adding subschemas.
        merged_schemas = resolve_subschemas(merged_schema, defs, visited_refs)
        return [
            merged(ms, rs)
            for subschema in schema["anyOf"]
            for rs in resolve_subschemas(subschema, defs, visited_refs)
            for ms in merged_schemas
        ]

    if "oneOf" in schema:
        # The "oneOf" keyword requires matching one of the subschemas, while _not_ matching
        # any of the others. Similarly to "not", the general case is out of scope.
        # Since many use cases are equivalent, we treat this keyword as an synonym of "anyOf".
        merged_schema = schema.copy()
        del merged_schema["oneOf"]
        # Resolve any other keywords ("allOf", "anyOf", ...) before adding subschemas.
        merged_schemas = resolve_subschemas(merged_schema, defs, visited_refs)
        return [
            merged(ms, rs)
            for subschema in schema["oneOf"]
            for rs in resolve_subschemas(subschema, defs, visited_refs)
            for ms in merged_schemas
        ]

    return [schema]


class UnknownSchemaTypeError(Exception):
    """
    Raised when a JSON schema doesn't contain a known type.
    """

    def __init__(self, schema):
        super().__init__(f"Unknown schema type for schema: {schema}")


# pylint: disable-next=invalid-name
def JsonSchemaAcceptor(schema, context=None):
    """
    Accept JSON according to a schema
    """
    if context is None:
        context = {"defs": defaultdict(dict), "path": ""}

    if schema.get("nullable"):
        non_nullable = schema.copy()
        del non_nullable["nullable"]
        return AnyOfAcceptor([{"type": "null"}, non_nullable], context)

    if "$defs" in schema:
        schema_defs = schema["$defs"]
        if "$id" in schema_defs:
            raise SchemaNotImplementedError("$defs.$id")
        for def_name, def_schema in schema_defs.items():
            # Not clear whether defs are relative or absolute, do both
            context["defs"][f"#/$defs{context['path']}/{def_name}"] = def_schema
            context["defs"][f"#/$defs/{def_name}"] = def_schema

    schemas = resolve_subschemas(schema, context["defs"], {})
    if len(schemas) == 1:
        schema = schemas[0]
    else:
        return AnyOfAcceptor(schemas, context)

    if "not" in schema:
        # Since this is for autoregressive generation, it doesn't make sense to get an LLM
        # to generate something in order to then negate its validity in retrospect. Thus,
        # supporting the "not" keyword in the general case is out of scope.
        # Some use cases of "not" could be handled, like rejecting generation of certain
        # values for primitive types, e.g. a boolean or string value.
        raise SchemaNotImplementedError("not")

    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        return AnyOfAcceptor(
            [merged(schema, {"type": type}) for type in schema_type], context
        )

    # Be robust with loosely-defined schemas
    if schema_type is None:
        if "properties" in schema:
            schema_type = "object"
        elif "items" in schema:
            schema_type = "array"

    if schema_type == "boolean":
        acceptor = BooleanAcceptor()
    elif schema_type == "number":
        acceptor = NumberSchemaAcceptor(schema)
    elif schema_type == "integer":
        acceptor = NumberSchemaAcceptor(schema)
    elif schema_type == "string":
        acceptor = StringSchemaAcceptor(schema)
    elif schema_type == "null":
        acceptor = NullAcceptor()
    elif "const" in schema:
        acceptor = ConstSchemaAcceptor(schema)
    elif "enum" in schema:
        acceptor = EnumSchemaAcceptor(schema)
    elif schema_type == "object":
        if "properties" in schema:
            # Only allows named properties in the object.
            # See comment in constructor.
            acceptor = ObjectSchemaAcceptor(schema, context)
        else:
            # Allows any properties in the object.
            acceptor = ObjectAcceptor()
    elif schema_type == "array":
        if "items" in schema:
            acceptor = ArraySchemaAcceptor(schema, context)
        else:
            acceptor = ArrayAcceptor()
    else:
        raise UnknownSchemaTypeError(schema)

    return acceptor


# pylint: disable-next=invalid-name
def EncapsulatedJsonSchemaAcceptor(schema):
    """
    Accept JSON according to a schema, where the JSON is part of a larger text
    and is delimited by starting line ```json and end line ```.
    """
    return SequenceAcceptor(
        [
            WaitForAcceptor(TextAcceptor("```json\n")),
            JsonSchemaAcceptor(schema),
            TextAcceptor("\n```"),
        ]
    )


class JsonSchemaAcceptorDriver:
    """
    Utility class to drive a JsonSchemaAcceptor
    """

    class TokenRejected(Exception):
        """
        Raised when the token cannot advance any of the current acceptors.
        """

    class CharacterRejected(Exception):
        """
        Raised in character-by-character mode (advance_char method) when the
        character cannot advance any of the current acceptors.
        """

    @classmethod
    def driver_factory_for_model(
        cls,
        vocabulary: Iterable[Tuple[int, str]],
        eos_id: int,
    ) -> callable:
        """
        Sets up the data structures (slow, one time) and returns a factory function
        that creates JsonSchemaAcceptorDriver objects for the model.
        """
        # Make sure the eos token is removed.
        # Ideally, the caller should not pass any special tokens (bos, eos, unk, pad, ...)
        vocabulary_list = [
            (token, fragment) for token, fragment in vocabulary if token != eos_id
        ]
        vocabulary_dict = dict(vocabulary_list)
        vocabulary_trie = TokenAcceptor.prepare_vocabulary(vocabulary_list)
        prepare_json_acceptor_tries(vocabulary_trie)

        def _factory(
            schema: dict, is_encapsulated_json: bool = False
        ) -> JsonSchemaAcceptorDriver:
            return JsonSchemaAcceptorDriver(
                vocabulary_dict, vocabulary_trie, eos_id, schema, is_encapsulated_json
            )

        return _factory

    def __init__(
        self,
        vocabulary_dict: dict[int, str],
        vocabulary_trie: TokenTrie,
        eos_id: int,
        schema: dict,
        is_encapsulated_json: bool = False,
    ):
        self.vocabulary = vocabulary_dict
        self.trie = vocabulary_trie
        self.eos_id = eos_id
        if is_encapsulated_json:
            self.acceptor = EncapsulatedJsonSchemaAcceptor(schema)
        else:
            self.acceptor = JsonSchemaAcceptor(schema)
        self.cursors = self.acceptor.get_cursors()
        self.debug_max_cursors = 0

    def in_accepted_state(self) -> bool:
        """
        Returns True if the acceptor has reached a valid final state.
        """
        return any(cursor.in_accepted_state() for cursor in self.cursors)

    def select_valid_tokens(self) -> int:
        """
        Given the current valid state(s) of the acceptor, return the tokens
        in the vocabulary that can advance the acceptor towards another
        valid state(s).
        """
        accepted_token_ids = TokenAcceptor.match_all(self.cursors, self.trie)
        if self.in_accepted_state():
            accepted_token_ids |= 1 << self.eos_id
        return accepted_token_ids

    def debug_select_valid_tokens(self, debug_output_fn=print) -> int:
        """
        Same as select_valid_tokens() but prints debug information.
        """
        accepted_token_ids = TokenAcceptor.debug_match_all(
            self.cursors, self.trie, debug_output_fn
        )
        if self.in_accepted_state():
            accepted_token_ids |= 1 << self.eos_id
        return accepted_token_ids

    def advance_token(self, token):
        """
        Advance the state(s) of the acceptor with the chosen token.
        """
        if token == self.eos_id:
            if not self.in_accepted_state():
                raise JsonSchemaAcceptorDriver.TokenRejected()
            self.cursors = []
            return
        fragment = self.vocabulary[token]
        cursors = self.cursors
        for char in fragment:
            cursors = TokenAcceptor.advance_all(cursors, char)
            if not cursors:
                raise JsonSchemaAcceptorDriver.TokenRejected()
        self.cursors = cursors

    def debug_advance_token(self, token, debug_output_fn=print):
        """
        Same as advance_token() but prints debug information.
        """
        if token == self.eos_id:
            if not self.in_accepted_state():
                raise JsonSchemaAcceptorDriver.TokenRejected()
            self.cursors = []
            return
        fragment = self.vocabulary[token]
        cursors = self.cursors
        for char in fragment:
            cursors = TokenAcceptor.advance_all(cursors, char)
            ncursors = len(cursors)
            if ncursors > self.debug_max_cursors:
                self.debug_max_cursors = ncursors
            debug_output_fn(f"ADVANCE char={repr(char)} cursors({ncursors})")
            if ncursors:
                debug_output_fn("  " + "\n  ".join(repr(c) for c in cursors))
            if ncursors == 0:
                raise JsonSchemaAcceptorDriver.TokenRejected()
        self.cursors = cursors

    def advance_char(self, char):
        """
        Advance the state(s) of the acceptor one character at a time.
        This is useful for applications such as partial JSON parsing.
        """
        cursors = TokenAcceptor.advance_all(self.cursors, char)
        if not cursors:
            raise JsonSchemaAcceptorDriver.CharacterRejected()
        self.cursors = cursors

    def get_current_value_paths(self):
        """
        This function will return the JSON path(s) for the value(s) that the
        last accepted character was attached to. More than one path means
        more than one possible parse for the JSON according to the schema,
        and it's rare.
        The value path is useful to know where in the JSON are we accepting a
        value (string, number, boolean, null) at the moment. An empty array
        means the last character was a syntactic elements (punctuation,
        whitespace) or part of the name of a property.
        Note that this function is mostly useful when advancing character by
        character, because a token can straddle across states and thus its
        path is not well-defined.
        """
        return [
           f"${path}"
           for path in set((
               cursor.get_value_path()
               for cursor in self.cursors if cursor.is_in_value()
           ))
        ]


