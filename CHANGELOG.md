# Changelog

Notable changes to  Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
## [Unreleased]

-->

## [0.6.0] - 20250526

### Changed

- Major overall refactoring
  - reorganize tool-calling logic, including in the `toolcall_mixin` class and `toolio.toolcall` nodule overall
    - New main entry class `toolio.local_model_runner` also has cleaned up iterative vs one-shot & tool-calling vs regular completion methods
  - Direct use of newer `mlx_lm` facilities such as `mlx_lm.sample_utils.make_sampler`, `mlx.generate_step_with_schema`, `mlx_lm.utils.GenerationResponse`, `mlx_lm.utils.stream_generate` and additions to `logits_processors`
    - See e.g. `llm_helper.Model.completion` & `llm_helper.Model.make_logit_bias_processor`. Logits bias via registered processors is now how we do the schema steering
  - vendored in `llm_structured_output` into Toolio (`toolio.vendor.llm_structured_output`)
  - Modularize handling of completion responses in its various forms (low-level Toolio, low-level mlx_lm, OpenAI-compatible) into `toolio.response_helper`
- Mechanism for JSON schema injection into prompts. See, for example `toolio.common.DEFAULT_JSON_SCHEMA_CUTOUT` & `toolio.common.replace_cutout`
- Accommodation of upstream changes & improved package installation logic

### Added

- `toolio.local_model_runner`, a new main encapsulation for running model as local MLX workloads
- Lots of other bits related to the major changes summarized above 

## [0.5.2] - 20241209

### Added

- `toolio.common.load_or_connect` convenience function
- `reddit_newsletter` multi-agent demo

### Changed

- Make the `{json_schema}` template "cutout" configurable, and change the default (to `#!JSON_SCHEMA!#`)

### Fixed

- Clean up how optional dependencies are handled
- Tool-calling prompting enhancements
- Clean up HTTP client & server interpretation of tool-calling & schemata

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
