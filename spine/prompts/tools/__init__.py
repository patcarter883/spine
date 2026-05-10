"""Tool-specific instruction templates.

These instructions teach agents how to use available tools effectively.
Each tool instruction follows the DeepAgents pattern of providing:
- What the tool does
- When to use it
- When NOT to use it
- How to use it effectively
"""

from __future__ import annotations

# =============================================================================
# FILESYSTEM TOOLS
# =============================================================================

FILESYSTEM_INSTRUCTIONS = """### Filesystem Tools

You have access to the filesystem through these tools:
- `read_file`: Read a file's contents
- `write_file`: Create or overwrite a file
- `edit_file`: Make targeted edits to a file
- `ls`: List directory contents
- `glob`: Find files matching a pattern
- `grep`: Search for text within files

#### Best Practices

**Before editing:**
1. ALWAYS read the file first to understand existing content
2. Check for existing patterns and conventions
3. Identify related files that may need updates

**When editing:**
1. Make minimal, focused changes
2. Preserve existing formatting and style
3. Update related documentation

**When creating new files:**
1. Follow existing directory structure
2. Match naming conventions in the codebase
3. Include appropriate imports and type hints
4. Add docstrings to public functions/classes

#### File Reading Strategy

Use pagination for large files:
```
read_file(path="large_file.py", offset=1, limit=100)  # First 100 lines
read_file(path="large_file.py", offset=101, limit=100)  # Next 100 lines
```

#### Search Strategy

Use `glob` to find files, `grep` to find content:
```
glob(pattern="**/*.py")  # Find all Python files
grep(pattern="def authenticate", path="src/")  # Find authentication functions
```
"""

# =============================================================================
# AGENT PROVIDER (External Coding Agent)
# =============================================================================

AGENT_PROVIDER_INSTRUCTIONS = """### External Coding Agent

You have access to an external coding agent (OpenCode, Codex, or Claude Code) for implementation tasks.

#### What It Does

The external agent can:
- Write and modify code
- Run tests and commands
- Create and edit files
- Search the codebase
- Execute shell commands

#### When to Use

Use the external agent for:
- **Implementation tasks** that require writing code
- **Multi-file changes** that need coordination
- **Refactoring** across multiple modules
- **Test writing** for new features
- **Documentation updates** that need context

#### When NOT to Use

Do NOT use the external agent for:
- **Analysis tasks** that just need reasoning
- **Planning tasks** that define what to do
- **Decision tasks** that require judgment
- **Clarification** about requirements

#### How to Use Effectively

**1. Be Specific**
Describe exactly what needs to be done:

```
✓ GOOD:
"Add a `validate_email` function to `src/utils/validators.py` that checks:
- Valid email format using regex
- Domain has valid MX records
- Returns (is_valid: bool, error_message: str | None)"

✗ BAD:
"Add email validation"
```

**2. Provide Context**
Include relevant information:

```
Context:
- Using Python 3.11
- Project uses pydantic for validation
- Existing validators in src/utils/validators.py follow this pattern:
  def validate_X(value: str) -> tuple[bool, str | None]
```

**3. Set Scope**
Specify which files/directories to work in:

```
Scope:
- Work within: src/auth/
- Modify: src/auth/login.py, src/auth/session.py
- Create: tests/test_auth/test_login.py
```

**4. Define Acceptance Criteria**
List specific testable requirements:

```
Acceptance criteria:
- [ ] Users can log in with email/password
- [ ] Invalid emails return clear error message
- [ ] Sessions expire after 24 hours
- [ ] Rate limiting: max 5 attempts per minute
- [ ] All tests pass
```

#### Example Delegation Prompt

```
Implement JWT-based authentication in the `src/auth/` directory.

Requirements:
1. Login endpoint: POST /auth/login
   - Accepts: {"email": str, "password": str}
   - Returns: {"token": str, "expires_at": int}
   - Validates credentials against database

2. Token validation middleware
   - Extracts Bearer token from Authorization header
   - Verifies signature and expiration
   - Injects user_id into request context

3. Logout endpoint: POST /auth/logout
   - Invalidates token
   - Returns success message

Files to create/modify:
- src/auth/jwt_handler.py (new)
- src/auth/middleware.py (new)
- src/auth/routes.py (modify)
- tests/test_auth/test_jwt.py (new)

Constraints:
- Use PyJWT library (already installed)
- Tokens expire after 24 hours
- Use HS256 algorithm
- Secret key from environment: JWT_SECRET_KEY

Acceptance criteria:
- [ ] Login returns valid JWT for correct credentials
- [ ] Invalid credentials return 401 with clear message
- [ ] Protected endpoints reject invalid/expired tokens
- [ ] Logout invalidates tokens server-side
- [ ] All tests pass
```

#### Checking Results

After the agent completes:
1. Review changed files
2. Run tests to verify functionality
3. Check for unexpected changes
4. Verify acceptance criteria are met

#### Error Handling

If the agent fails:
1. Read the error message carefully
2. Check if scope was too broad
3. Simplify the task and retry
4. Report blockers to the user
"""

# =============================================================================
# SHELL EXECUTION
# =============================================================================

SHELL_INSTRUCTIONS = """### Shell Execution

You have access to a sandboxed shell for running commands.

#### Allowed Commands

- `python` / `python3` - Run Python scripts
- `pytest` - Run tests
- `git` - Version control operations
- `pip` / `uv` - Package management
- `make` - Build automation
- `ls`, `cat`, `grep`, `find` - File inspection

#### When to Use

- Running tests
- Installing dependencies
- Git operations
- Build processes

#### When NOT to Use

- Making network requests (use appropriate tools)
- Long-running processes (use background mode)
- Commands that modify system state unexpectedly

#### Best Practices

1. Check `pwd` before relative paths
2. Use `--dry-run` flags when available
3. Redirect output to files for large results
4. Check exit codes
"""

# =============================================================================
# SUBAGENT DELEGATION
# =============================================================================

SUBAGENT_INSTRUCTIONS = """### Sub-Agent Delegation

You can delegate tasks to specialized sub-agents.

#### Available Sub-Agent Types

1. **research-agent**: For gathering information
2. **implementation-agent**: For coding tasks
3. **review-agent**: For code review
4. **test-agent**: For writing tests

#### Delegation Strategy

**DEFAULT: Single sub-agent**

Most tasks should use ONE sub-agent:
- "Research authentication patterns" → 1 research-agent
- "Implement login feature" → 1 implementation-agent

**Parallelize ONLY for clear separation:**

- "Compare auth approaches in Django vs FastAPI" → 2 research-agents
- "Implement auth for web and API separately" → 2 implementation-agents

#### Key Principles

- **Avoid premature decomposition**: Don't break "implement X" into multiple narrow tasks
- **Bias toward single agent**: One comprehensive task is more efficient
- **Parallelize for comparison**: When explicitly comparing different things

#### Parallel Execution Limits

- Maximum concurrent sub-agents: 3
- Make multiple task() calls in one response for parallelism
- Each sub-agent returns findings independently
"""

# =============================================================================
# TOOL REGISTRY
# =============================================================================

_TOOL_INSTRUCTIONS: dict[str, str] = {
    "filesystem": FILESYSTEM_INSTRUCTIONS,
    "agent_provider": AGENT_PROVIDER_INSTRUCTIONS,
    "shell": SHELL_INSTRUCTIONS,
    "subagent": SUBAGENT_INSTRUCTIONS,
    # Aliases
    "file": FILESYSTEM_INSTRUCTIONS,
    "agent": AGENT_PROVIDER_INSTRUCTIONS,
    "execute": SHELL_INSTRUCTIONS,
    "opencode": AGENT_PROVIDER_INSTRUCTIONS,
    "codex": AGENT_PROVIDER_INSTRUCTIONS,
    "claude_code": AGENT_PROVIDER_INSTRUCTIONS,
}


def get_tool_instructions(tool_name: str) -> str:
    """Get instructions for a specific tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool-specific instructions, or empty string if not found
    """
    return _TOOL_INSTRUCTIONS.get(tool_name, "")
