# T2S - Text to SQL CLI

[![PyPI version](https://badge.fury.io/py/t2s-cli.svg)](https://badge.fury.io/py/t2s-cli)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A powerful, **privacy-first** terminal-based Text-to-SQL converter that transforms natural language queries into SQL statements using state-of-the-art AI models - all running **locally** on your machine.

**Created by Lakshman Turlapati**

---

## 🔒 Privacy & Security First

**Your data never leaves your machine.** T2S runs entirely locally without any external APIs, ensuring complete data privacy and security. Perfect for sensitive enterprise databases and personal projects alike.

## ✨ Features

🤖 **Advanced AI Workflows**
- Multiple specialized AI workflows optimized for different use cases
- Intelligent model selection based on your hardware and accuracy needs
- Automatic schema analysis and query optimization
- Support for complex multi-table joins and aggregations

🎯 **Smart Database Integration**
- Auto-detection of local databases (SQLite, PostgreSQL, MySQL)
- Real-time schema analysis and intelligent table selection
- Query validation and automatic error correction
- Beautiful result visualization in terminal

🎨 **Beautiful Terminal Experience**
- Rich ASCII art branding and intuitive interface
- Interactive configuration with memory compatibility warnings
- Real-time loading animations and progress indicators
- Syntax highlighting for SQL queries and results

🛡️ **Enterprise-Ready**
- Complete offline operation - no internet required after setup
- Local model storage and management
- Cross-platform support (macOS, Windows, Linux)
- Memory-aware model recommendations

## 🤖 AI Model Recommendations

T2S offers multiple AI workflows, each optimized for different accuracy and performance needs:

### 🏆 **For Best Results (95%+ Accuracy)**
- **Gemma 3 (12B)** - Recommended for production use and complex queries
- **Defog SQLCoder (7B)** - Specialized SQL model, excellent for technical users

### ⚡ **For Most Users (80-90% Accuracy)**
- **Gemma 3 (4B)** - **Recommended for 95% of users** - Perfect balance of accuracy and performance
- Runs efficiently on most modern hardware with excellent results

### 🪶 **For Experimentation (40-60% Accuracy)**
- **SmolVLM (500M)** - Ultra-lightweight, runs on any computer
- Great for learning, testing, and resource-constrained environments

### 💡 **Choosing Your Model**
- **Most users:** Start with **Gemma 3 (4B)** - it provides the best balance
- **Power users:** Upgrade to **Gemma 3 (12B)** or **SQLCoder** for maximum accuracy
- **Experimenters:** Try **SmolVLM** for quick testing and learning

T2S automatically warns you about memory compatibility and helps you choose the right model for your system.

## 🚀 Installation

### Quick Start
```bash
pip install t2s-cli
```

### For Gemma Model Support
Some models may require additional dependencies:

```bash
# Option 1: Install with Gemma support
pip install t2s-cli[gemma]

# Option 2: For all features
pip install t2s-cli[all]
```

### macOS Users (if needed)
```bash
# If you encounter SentencePiece build issues
brew install sentencepiece protobuf
pip install t2s-cli
```

## 🎯 Quick Start

1. **Launch T2S**
   ```bash
   t2s
   ```

2. **First-time Setup** (Interactive Wizard)
   - Choose your AI model based on your needs and system capabilities
   - Connect to your databases (SQLite, PostgreSQL, MySQL)
   - Download your selected model (happens automatically)

3. **Start Querying**
   ```bash
   t2s query "Show me all customers who ordered in the last month"
   ```

## 💻 Usage Examples

### Interactive Mode
```bash
t2s
```
Launches the full interactive experience with model management, database configuration, and query execution.

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

## 🗄️ Supported Databases

- **SQLite** - Perfect for local development and small projects
- **PostgreSQL** - Enterprise-grade relational database
- **MySQL** - Popular web application database
- More databases coming soon!

## 🌟 Real-World Examples

### Basic Queries
```
Input: "Show me all active users"
Output: SELECT * FROM users WHERE status = 'active';

Input: "Count orders by month"
Output: SELECT DATE_FORMAT(created_at, '%Y-%m') as month, COUNT(*) 
        FROM orders GROUP BY month ORDER BY month;
```

### Complex Queries
```
Input: "Find customers who spent more than $1000 last quarter"
Output: SELECT c.name, c.email, SUM(o.total) as total_spent
        FROM customers c
        JOIN orders o ON c.id = o.customer_id
        WHERE o.created_at >= DATE_SUB(NOW(), INTERVAL 3 MONTH)
        GROUP BY c.id, c.name, c.email
        HAVING total_spent > 1000
        ORDER BY total_spent DESC;
```

## 🎓 Who Is This For?

- **Database Administrators** - Quickly explore and analyze database contents
- **Developers** - Rapid prototyping and database querying during development
- **Data Analysts** - Convert business questions into SQL without memorizing syntax
- **Students & Researchers** - Learn SQL and experiment with AI models locally
- **Privacy-Conscious Users** - Query sensitive data without sending it to external services

## 🛠️ Development & Contributing

We welcome contributions! This project has been crafted with significant effort and attention to detail.

### Setup Development Environment
```bash
git clone https://github.com/lakshmanturlapati/t2s-cli.git
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

## 🎯 Roadmap

- [ ] Additional database support (Oracle, SQL Server)
- [ ] Natural language result explanations
- [ ] Query history and favorites
- [ ] Custom model fine-tuning support
- [ ] Integration with popular BI tools

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project was developed with the support and guidance of the **University of Texas at Dallas**.

**Special Thanks:**
- **Professor Srikanth Kannan** - For invaluable guidance and shedding light on the path forward
- The open-source community for the excellent libraries that made this possible
- HuggingFace for providing the model infrastructure
- All contributors and users who help make T2S better

## 👨‍💻 Author

**Lakshman Turlapati**

- GitHub: [@lakshmanturlapati](https://github.com/lakshmanturlapati)
- Project Repository: [t2s-cli](https://github.com/lakshmanturlapati/t2s-cli)

---

**"Bringing the power of AI to your database, locally and securely."**

*This README will be updated as the project continues to evolve. Your feedback and contributions are always welcome!* 