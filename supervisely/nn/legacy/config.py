# coding: utf-8
import collections

from supervisely.io.json import load_json_file

from jsonschema import Draft4Validator, validators


# From https://python-jsonschema.readthedocs.io/en/v2.6.0/faq/
def _extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(prop, subschema["default"])

        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return validators.extend(validator_class, {"properties": set_defaults})


class MultiTypeValidator(object):
    DEFINITIONS = 'definitions'

    def __init__(self, schema_fpath):
        vtor_class = _extend_with_default(Draft4Validator)
        schemas = load_json_file(schema_fpath)
        # Detach common definitions from the named schemas and inline them in into every schema.
        definitions = schemas.pop(self.DEFINITIONS, {})
        for name, schema in schemas.items():
            schema.setdefault(self.DEFINITIONS, {}).update(definitions)

        self._validators = {name: vtor_class(schema) for name, schema in schemas.items()}

    def val(self, type_name, obj):
        validator = self._validators.get(type_name)
        if validator is None:
            raise RuntimeError('JSON validator is not defined. Type: {}'.format(type_name))
        try:
            validator.validate(obj)
        except Exception as e:
            raise RuntimeError('Error occurred during JSON validation. Type: {}. Exc: {}. See documentation'.format(
                type_name, str(e)
            )) from None  # suppress previous stacktrace, save all required info


class JsonConfigValidator(MultiTypeValidator):
    def __init__(self, schema_fpath=None):
        super().__init__(schema_fpath or '/workdir/src/schemas.json')

    def validate_train_cfg(self, config):
        # store all possible requirements in schema, including size % 16 etc
        self.val('training_config', config)

    def validate_inference_cfg(self, config):
        # store all possible requirements in schema
        self.val('inference_config', config)


class AlwaysPassingConfigValidator:
    @classmethod
    def validate_train_cfg(cls, config):
        pass

    @classmethod
    def validate_inference_cfg(cls, config):
        pass


def update_recursively(lhs, rhs):
    """Performs updating like lhs.update(rhs), but operates recursively on nested dictionaries."""
    for k, v in rhs.items():
        if isinstance(v, collections.Mapping):
            lhs[k] = update_recursively(lhs.get(k, {}), v)
        else:
            lhs[k] = v
    return lhs


def update_strict(lhs, rhs):
    for k, v in rhs.items():
        if k in lhs:
            raise ValueError('Key collision while strict updating. Key:"{}"'.format(str(k)))
        lhs[k] = v
    return lhs
