"""
Tests for Prometheus Skill Ecosystem.
"""

import pytest
from pathlib import Path
from prometheus.core import ContextEngine, ContextConfig
from prometheus.skill import SkillLoader, SkillRegistry, SkillMeta, Skill


class TestSkillLoader:
    """Test skill loading and parsing."""

    def test_parse_basic_skill(self):
        """Test parsing a basic SKILL.md file."""
        content = """---
name: test-skill
version: 1.0.0
description: A test skill
tags: [test, example]
provides: [testing]
---

# Test Skill

This is a test skill with instructions.
"""
        skill = SkillLoader.parse(content, Path("SKILL.md"))

        assert skill is not None
        assert skill.meta.name == "test-skill"
        assert skill.meta.version == "1.0.0"
        assert skill.meta.description == "A test skill"
        assert "test" in skill.meta.tags
        assert "testing" in skill.meta.provides
        assert "This is a test skill" in skill.content

    def test_parse_skill_with_requires(self):
        """Test parsing a skill with dependencies."""
        content = """---
name: dependent-skill
version: 2.0.0
description: Depends on other skills
tags: [advanced]
requires: [base-skill, util-skill]
provides: [advanced_feature]
---

Instructions here.
"""
        skill = SkillLoader.parse(content, Path("SKILL.md"))

        assert skill is not None
        assert "base-skill" in skill.meta.requires
        assert "util-skill" in skill.meta.requires

    def test_parse_invalid_skill_no_frontmatter(self):
        """Test parsing a skill without YAML frontmatter."""
        content = """# No frontmatter

Just a heading, no metadata.
"""
        skill = SkillLoader.parse(content, Path("SKILL.md"))
        assert skill is None

    def test_parse_invalid_skill_empty(self):
        """Test parsing empty content."""
        skill = SkillLoader.parse("", Path("SKILL.md"))
        assert skill is None

    def test_create_skill_template(self):
        """Test creating a skill template."""
        template = SkillLoader.create_skill_template(
            name="my-skill",
            description="Does something cool",
            instructions="Be helpful!",
            tags=["utility"],
        )

        assert "name: my-skill" in template
        assert "description: Does something cool" in template
        assert "Be helpful!" in template
        assert "utility" in template

    def test_load_from_dir(self, tmp_path):
        """Test loading skills from a directory."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Create two skill files
        (skills_dir / "skill1").mkdir()
        (skills_dir / "skill1" / "SKILL.md").write_text("""---
name: skill-1
version: 1.0.0
description: First skill
tags: [test]
---

Skill 1 content.
""")

        (skills_dir / "skill2").mkdir()
        (skills_dir / "skill2" / "SKILL.md").write_text("""---
name: skill-2
version: 2.0.0
description: Second skill
tags: [test, advanced]
---

Skill 2 content.
""")

        skills = SkillLoader.load_from_dir(skills_dir)
        assert len(skills) == 2

        names = {s.meta.name for s in skills}
        assert "skill-1" in names
        assert "skill-2" in names


class TestSkillRegistry:
    """Test skill registry operations."""

    @pytest.fixture
    def engine(self):
        """Create a ContextEngine for SkillRegistry."""
        config = ContextConfig()
        return ContextEngine(config)

    @pytest.fixture
    def registry(self, engine):
        """Create a SkillRegistry with engine."""
        return SkillRegistry(engine)

    def test_register_and_find(self, registry):
        """Test registering and finding skills."""

        skill = Skill(
            meta=SkillMeta(
                name="test-skill",
                version="1.0.0",
                description="Test",
                tags=["test"],
            ),
            instructions="Test instructions",
            source_path=Path("test/SKILL.md"),
        )
        registry.register(skill)

        found = registry.find_by_name("test-skill")
        assert len(found) == 1
        assert found[0].meta.name == "test-skill"

    def test_find_by_tag(self, registry):
        """Test finding skills by tag."""

        for i in range(3):
            skill = Skill(
                meta=SkillMeta(
                    name=f"skill-{i}",
                    version="1.0.0",
                    description="Test",
                    tags=["test"] if i < 2 else ["advanced"],
                ),
                instructions="Test",
                source_path=Path(f"skill-{i}/SKILL.md"),
            )
            registry.register(skill)

        test_skills = registry.find_by_tag("test")
        assert len(test_skills) == 2

        advanced_skills = registry.find_by_tag("advanced")
        assert len(advanced_skills) == 1

    def test_suggest_skills(self, registry):
        """Test skill suggestion."""

        # Register skills with different descriptions
        skills_data = [
            ("code-review", "Expert code review for Python and JavaScript", ["code"]),
            ("data-analysis", "Statistical analysis and data visualization", ["data"]),
            ("api-design", "RESTful API design and OpenAPI specs", ["api"]),
        ]

        for name, desc, tags in skills_data:
            skill = Skill(
                meta=SkillMeta(name=name, version="1.0.0", description=desc, tags=tags),
                instructions="Test",
                source_path=Path(f"{name}/SKILL.md"),
            )
            registry.register(skill)

        # Search for code-related skills
        suggestions = registry.suggest("review my Python code", top_k=3)
        assert len(suggestions) > 0

        # code-review should be first
        if suggestions:
            assert "code-review" in [s.id for s in suggestions]

    def test_dependency_resolution(self, registry):
        """Test resolving skill dependencies."""

        # Create a dependency chain: base -> mid -> top
        base = Skill(
            meta=SkillMeta(name="base", version="1.0.0", description="Base"),
            instructions="Base",
            source_path=Path("base/SKILL.md"),
        )
        mid = Skill(
            meta=SkillMeta(name="mid", version="1.0.0", description="Mid", requires=["base"]),
            instructions="Mid",
            source_path=Path("mid/SKILL.md"),
        )
        top = Skill(
            meta=SkillMeta(name="top", version="1.0.0", description="Top", requires=["mid", "base"]),
            instructions="Top",
            source_path=Path("top/SKILL.md"),
        )

        for skill in [base, mid, top]:
            registry.register(skill)

        deps = registry.resolve_dependencies(top)
        dep_names = [d.id for d in deps]
        assert "base" in dep_names
        assert "mid" in dep_names

    def test_get_full_prompt(self, registry):
        """Test generating the full skill prompt."""

        skill = Skill(
            meta=SkillMeta(
                name="helper",
                version="1.0.0",
                description="A helpful assistant skill",
                tags=["help"],
            ),
            instructions="You are a helpful assistant. Always be polite.",
            source_path=Path("helper/SKILL.md"),
        )
        registry.register(skill)

        prompt = registry.get_full_prompt(query="help me")
        assert "helper" in prompt
        assert "helpful assistant" in prompt.lower()

    def test_list_all(self, registry):
        """Test listing all skills."""

        for i in range(5):
            skill = Skill(
                meta=SkillMeta(
                    name=f"skill-{i}",
                    version="1.0.0",
                    description=f"Skill {i}",
                    tags=["test"],
                ),
                instructions="Test",
                source_path=Path(f"skill-{i}/SKILL.md"),
            )
            registry.register(skill)

        all_skills = registry.list_all()
        assert len(all_skills) == 5
