# Changelog

Notable changes to  Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
## [Unreleased]

-->

## [Unreleased]

### Added

- `toolio.common.load_or_connect` convenience function

### Changed

- Make the `{json_schema}` template "cutout" configurable, and change the default

## [0.5.1] - 20241029

### Added

- Demo `demo/re_act.py`
- `common.response_text()` function to simplify usage

### Fixed

- Usage pattern of KVCache

### Changed

- Decode `json_schema` if given as a string

### Removed

- `json_response` arg to `llm_helper.complete()`; just go by whether json_schema is None

## [0.5.0] - 20240903

### Added

- `llm_helper.debug_model_manager`—a way to extract raw prompt & schema/tool-call info for debugging of underlying LLM behavior
- docs beyond the README (`doc` folder)
- test cases
- demo/algebra_tutor.py
- demo/blind_obedience.py

### Changed

- use of logger rather than trace boolean, throughout
- further code modularizarion and reorg
- improvements to default prompting
- more elegant handling of install from an unsupported OS

### Fixed

- handling of multi-trip scenarios

## [0.4.2] - 20240807

### Added

- notes on how to override prompting

### Changed

- processing for function-calling system prompts

### Fixed

- server startup 😬

## [0.4.1] - 20240806

### Added

- demo `demo/zipcode.py`
- support for multiple workers & CORS headers (`--workers` & `--cors_origin` cmdline option)

### Fixed

- async tool definitions

## [0.4.0] - 20240802

### Added

- `toolio.responder` module, with coherent factoring from `server.py`
- `llm_helper.model_manager` convenience API for direct Python loading & inferencing over models
- `llm_helper.extract_content` helper to simplify the OpenAI-style streaming completion responses
- `test/quick_check.py` for quick assessment of LLMs in Toolio
- Mistral model type support

### Changed

- Turn off prompt caching until we figure out [#12](https://github.com/OoriData/Toolio/issues/12)
- Have responders return actual dicts, rather than label + JSON dump
- Factor out HTTP protocol schematics to a new module
- Handle more nuances of tool-calling tokenizer setup
- Harmonize tool definition patterns across invocation styles

### Fixed

- More vector shape mamagement

### Removed

- Legacy OpenAI-style function-calling support

## [0.3.1] - 20240722

### Added

- `trip_timeout` command line option for `toolio_request`
- Support for mixtral model type
- Model loading timing

### Fixed

- [AttributeError: 'ReusableKVCache' object has no attribute 'head_dim'](https://github.com/OoriData/Toolio/issues/10)

### Changed

- `timeout` client param to `trip_timeout`

## [0.3.0] - 20240717

### Added

- tool/param.rename, e.g. for tool params which are Python keywords or reserved words
- API example in README
- Type coercion for tool parameters
- Ability to rename params in for tools
- Three test cases, including currency conversion

### Fixed

- Excessive restrictions in OpenAI API

## [0.2.0] - 20240702

### Added

- A couple of test cases

### Fixed

- Error when tool is not used

## [0.1.0] - 20240701

- Initial standalone release candidate
