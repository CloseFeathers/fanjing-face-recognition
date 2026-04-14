# 贡献指南

感谢你对 Fanjing Face Recognition 的关注！本项目是 [FlowElement](https://github.com/FlowElement) 开源生态的一部分。

通用贡献规范（DCO 签署、PR 流程等）请参阅 [FlowElement 贡献指南](https://github.com/FlowElement/m_flow/blob/main/CONTRIBUTING.md)。

以下为本项目特有的开发说明。

## 开发环境搭建

```bash
git clone <repo-url>
cd fanjing-face-recognition
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-training.txt
```

## 代码风格

- 遵循 PEP 8，行宽 120 字符
- 使用类型注解（`typing` 模块）
- 变量名使用 snake_case，类名使用 PascalCase

## 报告问题

- 功能 Bug：通过 GitHub Issues 提交
- 安全漏洞：请参阅 [FlowElement 安全政策](https://github.com/FlowElement/m_flow/blob/main/SECURITY.md)
