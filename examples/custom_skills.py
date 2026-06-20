"""
Example: Creating Custom Skills
===============================

Demonstrates the skill ecosystem:
1. Create SKILL.md files
2. Load and register skills
3. Use skill suggestion and injection
"""

import asyncio
from pathlib import Path

from prometheus.core import ContextEngine, ContextConfig
from prometheus.skill import SkillLoader, SkillRegistry


def create_sample_skills(skills_dir: Path):
    """Create sample skill files for demonstration."""
    skills_dir.mkdir(parents=True, exist_ok=True)

    # Skill 1: Code Review
    code_review_dir = skills_dir / "code-review"
    code_review_dir.mkdir(exist_ok=True)
    (code_review_dir / "SKILL.md").write_text("""---
name: code-review
version: 1.0.0
description: Expert code review with security, performance, and style analysis
tags: [code, review, quality, security]
provides: [code_review, security_audit, style_check]
---

# Code Review Skill

You are an expert code reviewer. When reviewing code, follow these principles:

## Review Checklist
1. **Security**: Check for SQL injection, XSS, auth bypass, data leaks
2. **Performance**: Look for N+1 queries, memory leaks, inefficient algorithms
3. **Style**: Ensure consistent naming, proper error handling, clear comments
4. **Architecture**: Verify SOLID principles, proper separation of concerns
5. **Testing**: Check test coverage, edge cases, proper mocking

## Output Format
- Severity: CRITICAL / HIGH / MEDIUM / LOW
- File: line number
- Issue description
- Suggested fix with code example

## Example
```python
# BAD
query = f"SELECT * FROM users WHERE id = {user_input}"

# GOOD
query = "SELECT * FROM users WHERE id = %s"
cursor.execute(query, (user_input,))
```
""")

    # Skill 2: Data Analysis
    data_analysis_dir = skills_dir / "data-analysis"
    data_analysis_dir.mkdir(exist_ok=True)
    (data_analysis_dir / "SKILL.md").write_text("""---
name: data-analysis
version: 1.0.0
description: Statistical analysis and data visualization expert
tags: [data, analysis, statistics, visualization]
requires: []
provides: [statistical_analysis, data_visualization, report_generation]
---

# Data Analysis Skill

You are an expert data analyst. Follow this workflow:

## Analysis Pipeline
1. **Data Understanding**: Examine shape, types, missing values, distributions
2. **Cleaning**: Handle missing values, outliers, inconsistent formats
3. **Exploration**: Descriptive statistics, correlations, patterns
4. **Modeling**: Select appropriate statistical tests or ML models
5. **Visualization**: Create clear, informative charts
6. **Reporting**: Summarize findings with actionable insights

## Best Practices
- Always check assumptions before applying statistical tests
- Use confidence intervals, not just p-values
- Provide both technical and non-technical explanations
- Include code to reproduce all analyses
- Document data sources and limitations
""")

    # Skill 3: API Design
    api_design_dir = skills_dir / "api-design"
    api_design_dir.mkdir(exist_ok=True)
    (api_design_dir / "SKILL.md").write_text("""---
name: api-design
version: 1.0.0
description: RESTful and GraphQL API design expert following OpenAPI 3.1 spec
tags: [api, design, rest, graphql, openapi]
provides: [api_design, endpoint_planning, openapi_spec]
---

# API Design Skill

You are an expert API designer. Follow REST best practices:

## Design Principles
1. **Resource-oriented**: Use nouns, not verbs (`/users` not `/getUsers`)
2. **HTTP methods**: GET(read), POST(create), PUT(replace), PATCH(update), DELETE(remove)
3. **Status codes**: 200(OK), 201(Created), 400(Bad Request), 404(Not Found), 500(Server Error)
4. **Versioning**: URL-based (`/v1/users`) or header-based
5. **Pagination**: Cursor-based for large datasets
6. **Error format**: Consistent JSON error responses
7. **Documentation**: OpenAPI 3.1 specification

## Security
- Rate limiting on all endpoints
- Input validation using JSON Schema
- Authentication: JWT or OAuth2
- CORS configuration
- HTTPS only
""")

    print(f"Created 3 sample skills in {skills_dir}")


async def main():
    print("=" * 60)
    print("Prometheus Skills Ecosystem Example")
    print("=" * 60)

    # Create sample skills
    skills_dir = Path("./example_skills")
    create_sample_skills(skills_dir)

    # Initialize engine with skills directory
    config = ContextConfig(
        skill_dirs=[skills_dir],
    )
    engine = ContextEngine(config)
    await engine.initialize()

    # List loaded skills
    print(f"\n--- Loaded Skills ({len(engine.skill_registry.skills)}) ---")
    for skill in engine.skill_registry.skills.values():
        print(f"  • {skill.id}")
        print(f"    Tags: {', '.join(skill.meta.tags)}")
        print(f"    Provides: {', '.join(skill.meta.provides)}")
        print()

    # Suggest skills for queries
    queries = [
        "Review my Python code for security issues",
        "Analyze the sales data and find trends",
        "Design a REST API for user management",
    ]

    print("--- Skill Suggestions ---")
    for query in queries:
        print(f"\nQuery: '{query}'")
        suggestions = engine.skill_registry.suggest(query, top_k=3)
        for s in suggestions:
            print(f"  → {s.id} (score: relevant)")
        if not suggestions:
            print("  → No matching skills found")

    # Generate full prompt with skills
    print("\n--- Generated Skill Prompt (for code review query) ---")
    prompt = engine.skill_registry.get_full_prompt(query="code review security")
    print(prompt[:400] + "...\n")

    # Dependency resolution demo
    print("--- Dependency Resolution ---")
    code_review = engine.skill_registry.find_by_name("code-review")
    if code_review:
        deps = engine.skill_registry.resolve_dependencies(code_review[0])
        print(f"Skills needed to run 'code-review':")
        for dep in deps:
            print(f"  • {dep.id}")

    print(f"\n✓ Skills example complete!")


if __name__ == "__main__":
    asyncio.run(main())
