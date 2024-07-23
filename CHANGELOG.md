# Changelog

Notable changes to  Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
## [Unreleased]

-->

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
