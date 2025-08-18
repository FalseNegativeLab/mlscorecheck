# GitHub Workflows

This directory contains the modernized GitHub Actions workflows for the mlscorecheck project.

## Workflows Overview

### 1. **Build & Test** (`build.yml`)
**Triggers:** Push/PR to main/develop branches

**Features:**
- **Code Quality**: Linting with flake8
- **Multi-platform Testing**: Ubuntu, Windows, macOS
- **Multi-version Testing**: Python 3.10, 3.11, 3.12, 3.13
- **Coverage Reporting**: Integrated with Codecov
- **Build Artifacts**: Uploads wheel and source distribution
- **Concurrency Control**: Cancels redundant runs

**Jobs:**
1. `lint` - Code quality checks
2. `test` - Cross-platform testing with coverage
3. `build` - Package building and artifact upload

### 2. **Release** (`release.yml`)
**Triggers:** GitHub releases or manual dispatch

**Features:**
- **Trusted Publishing**: Secure PyPI publishing without tokens
- **Test PyPI Support**: Option to publish to test environment
- **Release Validation**: Runs tests before publishing
- **GitHub Assets**: Automatically uploads wheels to releases
- **Dual Environment**: Supports both PyPI and Test PyPI

**Jobs:**
1. `validate-release` - Pre-publish validation
2. `build-and-publish` - Build and publish to PyPI
3. `create-github-release-assets` - Upload to GitHub releases

### 3. **Documentation** (`docs.yml`)
**Triggers:** Changes to docs/ or mlscorecheck/ directories

**Features:**
- **Sphinx Documentation**: Builds HTML documentation
- **Warning Detection**: Fails on warnings/errors
- **Artifact Upload**: Stores built docs as artifacts
- **Path-based Triggers**: Only runs when relevant files change

**Jobs:**
1. `build-docs` - Build and validate documentation

## Key Improvements

### Security
- **Trusted Publishing**: Uses OIDC tokens instead of API keys
- **Environment Protection**: Release jobs require approval
- **Minimal Permissions**: Only grants necessary permissions

### Performance
- **Concurrency Control**: Cancels redundant workflow runs
- **Conditional Execution**: Docs only build when needed
- **Artifact Caching**: Efficient dependency management

### Maintainability
- **Modern Actions**: Uses latest action versions (v4, v5)
- **Clear Job Names**: Descriptive and organized
- **Fail-Fast Strategy**: Quick feedback on failures
- **Comprehensive Testing**: Multiple OS and Python versions

### Developer Experience
- **Manual Dispatch**: Test releases manually
- **Clear Feedback**: Detailed job names and status
- **Artifact Downloads**: Easy access to build outputs
- **Path-based Triggers**: Reduced unnecessary runs

## Migration Notes

The new workflows replace the previous files:
- ❌ `pythonpackage.yml` → ✅ `build.yml`
- ❌ `pythonpublish.yml` → ✅ `release.yml`
- ❌ `codecov.yml` → ✅ Integrated into `build.yml`

All functionality is preserved while adding modern best practices and improved organization.
