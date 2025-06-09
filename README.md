
# T2S - Text to SQL CLI

[![PyPI version](https://img.shields.io/pypi/v/t2s-cli.svg)](https://pypi.org/project/t2s-cli/)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)

A powerful, privacy-first terminal-based Text-to-SQL converter that transforms natural language queries into SQL statements using state-of-the-art AI models—all running locally on your machine. Because your database deserves the best, and so does your privacy.

Created by Lakshman Turlapati

## Privacy & Security First

Your data stays where it belongs—on your machine. T2S operates entirely locally, free from external APIs, delivering uncompromised privacy and security. Ideal for sensitive enterprise databases and personal projects alike.

## Features

T2S is packed with capabilities to streamline your database querying experience:

### Advanced AI Workflows
* Multiple specialized AI workflows tailored to diverse use cases
* Intelligent model selection based on your hardware and accuracy requirements
* Automatic schema analysis and query optimization
* Support for complex multi-table joins and aggregations

### Smart Database Integration
* Auto-detection of local databases (SQLite, PostgreSQL, MySQL)
* Real-time schema analysis and intelligent table selection
* Query validation and automatic error correction
* Elegant result visualization in the terminal

### Beautiful Terminal Experience
* Rich ASCII art branding paired with an intuitive interface
* Interactive configuration with memory compatibility warnings
* Real-time loading animations and progress indicators
* Syntax highlighting for SQL queries and results

### Enterprise-Ready
* Fully offline operation—no internet required post-setup
* Local model storage and management
* Cross-platform support (macOS, Windows, Linux)
* Memory-aware model recommendations

## AI Model Recommendations

Choosing the right AI model is key to unlocking T2S’s full potential. Here’s how to pick one that fits your needs:

### For Best Results (95%+ Accuracy)
* **Gemma 3 (12B)** - Ideal for production environments and complex queries
* **Defog SQLCoder (7B)** - A specialized SQL model, perfect for technical users

### For Most Users (80-90% Accuracy)
* **Gemma 3 (4B)** - Recommended for 95% of users, striking a balance between accuracy and performance
    * Runs efficiently on most modern hardware with impressive results

### For Experimentation (40-60% Accuracy)
* **SmolVLM (500M)** - Ultra-lightweight and runs on any computer
    * Great for learning, testing, and resource-limited setups

T2S includes memory compatibility warnings to guide you toward the best model for your system.

## Installation

### Quick Start
```bash
pip install t2s-cli
````

### For Gemma Model Support

Some models require additional dependencies:

```bash
# Option 1: Install with Gemma support
pip install t2s-cli[gemma]

# Option 2: For all features
pip install t2s-cli[all]
```

### macOS Users (if needed)

```bash
# If SentencePiece build issues arise
brew install sentencepiece protobuf
pip install t2s-cli
```

## Getting Started

### Launch T2S

```bash
t2s
```

### First-Time Setup (Interactive Wizard)

1.  Select an AI model based on your needs and system capabilities
2.  Connect to your databases (SQLite, PostgreSQL, MySQL)
3.  Download your chosen model (handled automatically)

### Start Querying

```bash
t2s query "Show me all customers who ordered in the last month"
```

## Usage Examples

### Interactive Mode

```bash
t2s
```

Launches an interactive experience with model management, database configuration, and query execution.

### Direct Query Mode

```bash
t2s query "Find top 5 products by revenue this year"
```

### Configuration Management

```bash
t2s config    # Full configuration menu
t2s models    # Manage AI models
t2s databases # Manage database connections
```

## Supported Databases

  * SQLite - Perfect for local development and small projects
  * PostgreSQL - Enterprise-grade relational database
  * MySQL - Popular choice for web applications

More database support on the way\!

## Real-World Examples

### Basic Queries

**Input:** "Show me all active users"
**Output:**

```sql
SELECT * FROM users WHERE status = 'active';
```

**Input:** "Count orders by month"
**Output:**

```sql
SELECT DATE_FORMAT(created_at, '%Y-%m') as month, COUNT(*)
FROM orders GROUP BY month ORDER BY month;
```

### Complex Queries

**Input:** "Find customers who spent more than $1000 last quarter"
**Output:**

```sql
SELECT c.name, c.email, SUM(o.total) as total_spent
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.created_at >= DATE_SUB(NOW(), INTERVAL 3 MONTH)
GROUP BY c.id, c.name, c.email
HAVING total_spent > 1000
ORDER BY total_spent DESC;
```

## Who Is This For?

T2S caters to a broad audience:

  * **Database Administrators** - Swiftly explore and analyze database contents
  * **Developers** - Prototype and query databases during development
  * **Data Analysts** - Turn business questions into SQL without syntax struggles
  * **Students & Researchers** - Learn SQL and experiment with AI models locally
  * **Privacy-Conscious Users** - Handle sensitive data without external exposure

## Development & Contributing

Contributions are welcome\! This project reflects significant effort and meticulous care.

### Setup Development Environment

```bash
git clone [https://github.com/lakshmanturlapati/t2s-cli.git](https://github.com/lakshmanturlapati/t2s-cli.git)
cd t2s-cli
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Quality

```bash
black .
flake8 .
```

## Roadmap

  * Support for additional databases (Oracle, SQL Server)
  * Natural language explanations for results
  * Query history and favorites
  * Custom model fine-tuning
  * Integration with popular BI tools

## License

This project is licensed under the MIT License—see the `LICENSE` file for details.

## Acknowledgments

Developed with support and guidance from the University of Texas at Dallas.

### Special Thanks:

  * Professor Srikanth Kannan - For invaluable guidance and inspiration
  * The open-source community for essential libraries
  * HuggingFace for model infrastructure
  * Contributors and users who help refine T2S

## Author

**Lakshman Turlapati**

  * GitHub: [@lakshmanturlapati](https://www.google.com/search?q=https://github.com/lakshmanturlapati)
  * Project Repository: [t2s-cli](https://www.google.com/search?q=https://github.com/lakshmanturlapati/t2s-cli)
  * Portfolio: [https://www.audienclature.com](https://www.audienclature.com)
  * LinkedIn: [https://www.linkedin.com/in/lakshman-turlapati-3091aa191/](https://www.linkedin.com/in/lakshman-turlapati-3091aa191/)

"Bringing the power of AI to your database, locally and securely."

This README will evolve with the project. Your feedback and contributions are always appreciated\!


