# GitHub Actions Workflows for T2S CLI

This directory contains GitHub Actions workflows for automating the build, test, and release process of the T2S CLI package.

## Workflows Overview

### 1. Continuous Integration (`ci.yml`)
**Triggers:** Push to main/master/develop, Pull Requests
**Purpose:** Run tests, linting, and security scans on every code change

**Jobs:**
- **Test**: Runs on Python 3.9-3.12 across Ubuntu, Windows, and macOS
- **Lint**: Code quality checks with Black, isort, flake8, and mypy
- **Security**: Vulnerability scanning with safety and bandit

### 2. Build and Publish (`build-and-publish.yml`)
**Triggers:** Push to main, Pull Requests, Releases
**Purpose:** Comprehensive testing and publishing pipeline

**Jobs:**
- **Test**: Cross-platform testing with AI model dependencies
- **Build**: Package building and validation
- **Publish-test**: Auto-publish to Test PyPI on main branch pushes
- **Publish**: Publish to PyPI on releases
- **Security-scan**: Additional security checks
- **Code-quality**: Extended code quality analysis

### 3. Release (`release.yml`)
**Triggers:** GitHub releases, Manual workflow dispatch
**Purpose:** Streamlined release and publishing process

**Jobs:**
- **Build**: Create source distribution and wheel
- **Publish-to-testpypi**: Optional Test PyPI deployment
- **Publish-to-pypi**: Production PyPI deployment
- **Create-github-release**: Automated GitHub release creation

## Setup Instructions

### 1. Repository Secrets

You need to configure these secrets in your GitHub repository:

#### For PyPI Publishing:
Go to **Settings → Secrets and variables → Actions** and add:

- `PYPI_API_TOKEN`: Your PyPI API token
- `TEST_PYPI_API_TOKEN`: Your Test PyPI API token

#### Getting API Tokens:

**PyPI Token:**
1. Go to https://pypi.org/manage/account/token/
2. Create a new API token with scope "Entire account"
3. Copy the token (starts with `pypi-`)

**Test PyPI Token:**
1. Go to https://test.pypi.org/manage/account/token/
2. Create a new API token with scope "Entire account"  
3. Copy the token (starts with `pypi-`)

### 2. Environment Protection (Optional but Recommended)

For additional security, set up environment protection:

1. Go to **Settings → Environments**
2. Create environments: `pypi` and `testpypi`
3. Add required reviewers for production deployments
4. Configure environment secrets if you prefer environment-specific tokens

### 3. Branch Protection

Consider setting up branch protection rules:

1. Go to **Settings → Branches**
2. Add rule for `main` branch
3. Require status checks to pass before merging
4. Select relevant workflow checks

## Usage

### Running Tests
Tests run automatically on every push and pull request. The CI workflow will:
- Test across multiple Python versions and operating systems
- Install AI model dependencies (including sentencepiece for Gemma models)
- Verify package imports and CLI functionality
- Run code quality and security checks

### Publishing Releases

#### Method 1: GitHub Releases (Recommended)
1. Go to **Releases** in your GitHub repository
2. Click **Create a new release**
3. Choose a tag version (e.g., `v0.1.1`)
4. Add release notes
5. Click **Publish release**
6. The package will automatically be built and published to PyPI

#### Method 2: Manual Workflow Dispatch
1. Go to **Actions** tab
2. Select "Release and Publish" workflow
3. Click **Run workflow**
4. Choose options (Test PyPI vs PyPI)
5. Click **Run workflow**

### Testing Releases
To test your package before production release:
1. Use the manual workflow dispatch with "Deploy to Test PyPI" checked
2. Install from Test PyPI: `pip install --index-url https://test.pypi.org/simple/ t2s-cli`
3. Test functionality, then proceed with production release

## Workflow Features

### AI Model Dependencies
The workflows handle the special requirements for AI models:
- **macOS**: Installs sentencepiece and protobuf via Homebrew
- **Ubuntu**: Installs build-essential for compilation
- **Windows**: Uses pre-built wheels when available

### Caching
- pip dependencies are cached to speed up builds
- Cache keys include Python version and pyproject.toml hash

### Error Handling
- Security scans continue on error (won't fail the build)
- Type checking continues on error (mypy can be strict)
- Build artifacts are uploaded even if some steps fail

### Cross-Platform Testing
All workflows test across:
- **Operating Systems**: Ubuntu, Windows, macOS
- **Python Versions**: 3.9, 3.10, 3.11, 3.12
- **Dependencies**: Both minimal and full installations

## Troubleshooting

### Common Issues

**1. Package Import Errors**
- Ensure all dependencies are in `pyproject.toml`
- Check that the package structure is correct
- Verify `__init__.py` files are present

**2. PyPI Upload Failures**
- Check API tokens are correct and not expired
- Ensure version numbers are unique (PyPI doesn't allow overwrites)
- Verify package name availability

**3. AI Model Dependencies**
- sentencepiece build failures on macOS: Ensure Homebrew dependencies are installed
- Large dependency download timeouts: Consider dependency caching strategies

**4. Test Failures**
- Check Python version compatibility
- Verify all test dependencies are installed
- Review test output for specific error details

### Getting Help

If you encounter issues:
1. Check the **Actions** tab for detailed error logs
2. Review individual workflow step outputs
3. Compare successful runs with failed ones
4. Consider running workflows manually to test fixes

## Monitoring

Keep an eye on:
- **Build times**: Should remain reasonable as dependencies grow
- **Test coverage**: Ensure new features have appropriate tests
- **Security alerts**: Address dependency vulnerabilities promptly
- **PyPI statistics**: Monitor download counts and user feedback 