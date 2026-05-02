# Modular Provider Interfaces - Design Specification

## Core Concept

All external services plug into SPINE via a unified provider interface. Providers are discovered, configured, and managed dynamically.

---

## 1. Provider Architecture

### 1.1 Base Provider Interface

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ProviderConfig:
    name: str
    type: str
    enabled: bool = True
    priority: int = 0  # For fallback ordering
    config: Dict[str, Any] = None

class Provider(ABC):
    """Base interface for all providers"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name
        self.type = config.type
        self._initialized = False
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Initialize provider with configuration"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Health check and configuration validation"""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Cleanup resources"""
        pass
    
    @property
    def is_available(self) -> bool:
        return self._initialized and self.validate()
```

### 1.2 Provider Registry

```python
class ProviderRegistry:
    """Central registry for all providers"""
    
    def __init__(self):
        self._providers: Dict[str, Provider] = {}
        self._factories: Dict[str, Type[Provider]] = {}
        
    def register_factory(self, provider_type: str, factory: Type[Provider]):
        """Register a provider factory"""
        self._factories[provider_type] = factory
    
    def load_providers(self, config_path: str = "spine.yaml"):
        """Load providers from config"""
        config = load_yaml(config_path)
        for provider_type, instances in config.get("providers", {}).items():
            for instance in instances:
                self.load_provider(provider_type, ProviderConfig(**instance))
    
    def load_provider(self, provider_type: str, config: ProviderConfig):
        """Instantiate and register a provider"""
        factory = self._factories.get(provider_type)
        if not factory:
            raise ValueError(f"No factory for provider type: {provider_type}")
        
        provider = factory(config)
        provider.configure(config.config or {})
        self._providers[config.name] = provider
        
    def get(self, name: str) -> Optional[Provider]:
        return self._providers.get(name)
    
    def get_by_type(self, provider_type: str) -> List[Provider]:
        return [p for p in self._providers.values() 
                if p.type == provider_type and p.is_available]
```

---

## 2. LLM Provider Interface

```python
class LLMResponse:
    """Standardized LLM response"""
    content: str
    usage: Dict[str, int]  # tokens
    finish_reason: str
    model: str
    request_id: str

class LLMProvider(Provider):
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> LLMResponse:
        pass
    
    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system_prompt: str = None,
        **kwargs
    ) -> AsyncIterator[str]:
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

# Built-in implementations
class OpenAIProvider(LLMProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = None
        
    def configure(self, config: Dict[str, Any]):
        import openai
        self.client = openai.AsyncOpenAI(api_key=config.get("api_key"))
        self.model = config.get("model", "gpt-4o")
        
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self._format_messages(prompt, kwargs.get("system_prompt")),
            **kwargs
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            usage=response.usage.dict(),
            finish_reason=response.choices[0].finish_reason,
            model=response.model,
            request_id=response.id
        )
```

---

## 3. Memory Provider Interface

```python
@dataclass
class MemoryEntry:
    key: str
    value: Any
    metadata: Dict[str, Any]
    timestamp: datetime
    ttl: Optional[int] = None

class MemoryProvider(Provider):
    """Persistent and ephemeral memory storage"""
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry, return reference ID"""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value by key"""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Semantic/fuzzy search"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete entry by key"""
        pass
    
    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern"""
        pass

# Example: Vector memory provider
class VectorMemoryProvider(MemoryProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = None
        
    def configure(self, config: Dict[str, Any]):
        import chromadb
        self.client = chromadb.PersistentClient(path=config.get("path", ".spine/vectors"))
        self.collection = self.client.get_or_create_collection("spine_memory")
        
    async def store(self, entry: MemoryEntry) -> str:
        self.collection.add(
            ids=[entry.key],
            documents=[str(entry.value)],
            metadatas=[entry.metadata]
        )
        return entry.key
        
    async def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        results = self.collection.query(query_texts=[query], n_results=limit)
        return [
            MemoryEntry(
                key=ids[i],
                value=docs[i],
                metadata=metadatas[i],
                timestamp=datetime.fromisoformat(metadatas[i].get("timestamp"))
            )
            for i in range(len(ids))
        ]
```

---

## 4. Tools Provider Interface (MCP Integration)

```python
class ToolCall:
    """Represents a tool invocation"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0

class ToolsProvider(Provider):
    """Dynamic tool loading, primarily via MCP"""
    
    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """Return available tools with schemas"""
        pass
    
    @abstractmethod
    async def invoke(self, call: ToolCall) -> ToolCall:
        """Execute tool and return result"""
        pass
    
    @abstractmethod
    def has_tool(self, name: str) -> bool:
        pass

# MCP Server Provider
class MCPProvider(ToolsProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.session = None
        self.tools = []
        
    def configure(self, config: Dict[str, Any]):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        self.server_params = StdioServerParameters(
            command=config.get("command", "npx"),
            args=config.get("args", []),
            env=config.get("env")
        )
        # Initialize connection
        
    async def list_tools(self) -> List[Dict[str, Any]]:
        if not self.session:
            await self._connect()
        tools = await self.session.list_tools()
        return [t.dict() for t in tools]
        
    async def invoke(self, call: ToolCall) -> ToolCall:
        if not self.session:
            await self._connect()
        try:
            result = await self.session.call_tool(call.tool_name, call.arguments)
            call.result = result
            call.error = None
        except Exception as e:
            call.error = str(e)
        return call
```

---

## 5. Storage Provider Interface

```python
class StorageProvider(Provider):
    """File/blob storage abstraction"""
    
    @abstractmethod
    async def read(self, path: str) -> bytes:
        pass
    
    @abstractmethod
    async def write(self, path: str, data: bytes) -> str:
        """Return URI or reference"""
        pass
    
    @abstractmethod
    async def delete(self, path: str) -> bool:
        pass
    
    @abstractmethod
    async def list(self, prefix: str = "") -> List[str]:
        pass
    
    @abstractmethod
    async def exists(self, path: str) -> bool:
        pass

# Git Provider for repo-native storage
class GitStorageProvider(StorageProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.repo = None
        
    def configure(self, config: Dict[str, Any]):
        import git
        self.repo = git.Repo(config.get("path", "."))
        
    async def write(self, path: str, data: bytes) -> str:
        # Write to working tree
        with open(path, "wb") as f:
            f.write(data)
        
        # Auto-commit with conventional commit message
        self.repo.index.add([path])
        self.repo.index.commit(f"chore: spine checkpoint {path}")
        
        return path
```

---

## 6. Notification Provider Interface

```python
@dataclass
class Notification:
    level: str  # info, warning, error, success
    title: str
    message: str
    details: Dict[str, Any] = None
    actions: List[Dict[str, str]] = None  # Button actions

class NotifyProvider(Provider):
    """Multi-channel notifications"""
    
    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        pass
    
    @abstractmethod
    async def ask(self, question: str, options: List[str]) -> str:
        """Request human input"""
        pass

# Slack/Discord integration
class DiscordNotifyProvider(NotifyProvider):
    async def send(self, notification: Notification) -> bool:
        webhook_url = self.config.config.get("webhook_url")
        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json={
                "embeds": [{
                    "title": notification.title,
                    "description": notification.message,
                    "color": self._level_to_color(notification.level)
                }]
            })
        return True
```

---

## 7. Plugin System

### 7.1 Plugin Manifest

```yaml
# spine-plugin.yaml
name: "spine-gcp"
version: "1.0.0"
description: "Google Cloud providers for SPINE"
providers:
  - type: "llm"
    class: "spine_gcp.GCPProvider"
    config_schema:
      project_id: string
      location: string
  - type: "storage"
    class: "spine_gcp.GCSStorageProvider"
integrations:
  - mcp: "spine-gcp-mcp"
```

### 7.2 Plugin Loader

```python
class PluginLoader:
    def __init__(self, plugin_dirs: List[str] = None):
        self.plugin_dirs = plugin_dirs or ["./spine-plugins"]
        self.registry = ProviderRegistry()
        
    def discover_plugins(self):
        """Find and load all plugins"""
        for plugin_dir in self.plugin_dirs:
            manifest_path = Path(plugin_dir) / "spine-plugin.yaml"
            if manifest_path.exists():
                self._load_plugin(manifest_path)
                
    def _load_plugin(self, manifest_path: Path):
        manifest = load_yaml(manifest_path)
        for provider_def in manifest.get("providers", []):
            # Dynamic import
            module_path, class_name = provider_def["class"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)
            
            # Register factory
            self.registry.register_factory(
                provider_def["type"], 
                provider_class
            )
```

---

## 8. Configuration Schema

```yaml
# spine.yaml
providers:
  llm:
    - name: primary
      type: openai
      config:
        api_key: ${OPENAI_API_KEY}
        model: gpt-4o
      priority: 1
      
    - name: local
      type: ollama
      config:
        host: localhost:11434
        model: qwen3:32b
      priority: 2
      
  memory:
    - name: session
      type: sqlite
      config:
        path: .spine/sessions.db
        
    - name: longterm
      type: vector
      config:
        path: .spine/vectors
        
  tools:
    - name: mcp-browser
      type: mcp
      config:
        command: npx
        args: ["@modelcontextprotocol/server-playwright"]
        
plugins:
  - name: aws
    source: github:spine-plugins/aws
    config:
      region: us-east-1

defaults:
  llm: primary
  memory: session
```

---

## 9. Conflict Resolution for Provider Results

When multiple providers return conflicting results, SPINE needs strategies to reconcile or escalate:

### 9.1 Conflict Detection

```python
@dataclass
class ConflictResult:
    key: str  # What conflicted (e.g., "database_choice", "tech_stack")
    values: Dict[str, Any]  # provider_name -> value
    confidence: Dict[str, float]  # provider_name -> confidence score
    metadata: Dict[str, Any]
```

### 9.2 Resolution Strategies

```python
class ConflictResolver:
    """Resolves conflicts between provider results"""
    
    STRATEGIES = {
        "confidence_weighted": self._confidence_weighted,
        "voting": self._voting,
        "consensus": self._consensus,
        "highest_priority": self._highest_priority,
        "human_escalate": self._human_escalate,
    }
    
    def resolve(
        self, 
        conflict: ConflictResult, 
        strategy: str = "confidence_weighted"
    ) -> Any:
        """Apply resolution strategy to conflict"""
        return self.STRATEGIES[strategy](conflict)
    
    def _confidence_weighted(self, conflict: ConflictResult) -> Any:
        """Weight results by provider confidence scores"""
        weighted = {
            k: v * conflict.confidence[k] 
            for k, v in conflict.values.items()
        }
        return max(weighted.items(), key=lambda x: x[1])[0]
    
    def _voting(self, conflict: ConflictResult) -> Any:
        """Simple majority vote among providers"""
        counts = Counter(conflict.values.values())
        winner, count = counts.most_common(1)[0]
        if count > len(conflict.values) / 2:
            return winner
        return self._confidence_weighted(conflict)  # Fallback
    
    def _consensus(self, conflict: ConflictResult) -> Any:
        """Require all providers to agree, otherwise escalate"""
        unique_values = set(conflict.values.values())
        if len(unique_values) == 1:
            return list(unique_values)[0]
        raise ConflictRequiresHuman(f"No consensus on {conflict.key}")
    
    def _highest_priority(self, conflict: ConflictResult) -> Any:
        """Trust highest priority provider's result"""
        return conflict.values.get(max(conflict.confidence.keys(), 
                                       key=lambda k: conflict.confidence[k]))
```

### 9.3 LLM Provider Confidence Scoring

```python
class LLMProviderWithConfidence(LLMProvider):
    async def generate_with_confidence(
        self, 
        prompt: str, 
        **kwargs
    ) -> Tuple[LLMResponse, float]:
        """Return response with self-reported confidence"""
        response = await self.generate(prompt, **kwargs)
        
        # Request confidence from the model
        confidence_prompt = f"""Rate your confidence (0.0-1.0) in this answer:
        Question: {prompt}
        Answer: {response.content}
        
        Return only a number:"""
        
        conf_response = await self.generate(confidence_prompt, max_tokens=10)
        try:
            confidence = float(conf_response.content.strip())
        except ValueError:
            confidence = 0.8  # Default
            
        return response, min(1.0, max(0.0, confidence))
```

### 9.4 Configuration

```yaml
# spine.yaml - conflict resolution config
conflict_resolution:
  default_strategy: confidence_weighted
  strategies_by_type:
    tech_decisions: voting
    architecture: consensus  # Critical - require agreement
    implementation_details: confidence_weighted
    factual_questions: highest_priority  # Trust most authoritative source
  
  escalation:
    threshold: 0.3  # If confidence spread > 0.3, ask human
    notify_channel: slack#agent-questions
```

---

## 10. Swarm Agent Provider Interface

### 10.1 Agent Role Definitions

```python
class AgentRole:
    """Defines specialized agent capabilities for swarm execution"""
    
    ROLES = {
        # Planning Phase Agents
        "explorer": {
            "capabilities": ["parse", "analyze", "summarize"],
            "description": "Analyzes requirements and identifies constraints"
        },
        "sme": {
            "capabilities": ["research", "search", "analyze_patterns"],
            "description": "Subject Matter Expert - domain research"
        },
        "planner": {
            "capabilities": ["draft", "synthesize", "decompose"],
            "description": "Creates and refines execution plans"
        },
        "critic": {
            "capabilities": ["review", "challenge", "drift_verify"],
            "description": "Reviews plans and validates work"
        },
        
        # Execution Phase Agents
        "coder": {
            "capabilities": ["implement", "write_code", "refactor"],
            "description": "Implements code changes"
        },
        "reviewer": {
            "capabilities": ["review", "security_check", "validate"],
            "description": "Code review and quality assurance"
        },
        "test_engineer": {
            "capabilities": ["write_tests", "run_tests", "verify"],
            "description": "Test creation and execution"
        },
        "designer": {
            "capabilities": ["design_ui", "specify_ux", "prototype"],
            "description": "UI/UX design specifications"
        },
    }

@dataclass
class AgentAssignment:
    """Assignment of work to an agent"""
    agent_id: str
    agent_role: str
    task_id: str
    capability: str
    input: Any
    exclusive_paths: List[str] = None  # File reservations
    
class SwarmAgentProvider(Provider):
    """Specialized agent provider for swarm execution"""
    
    @abstractmethod
    async def execute_task(self, assignment: AgentAssignment) -> Dict[str, Any]:
        """Execute a task with the assigned agent"""
        pass
    
    @abstractmethod
    async def coordinate_agents(self, 
                                 assignments: List[AgentAssignment]) -> List[Dict[str, Any]]:
        """Execute multiple agent assignments with conflict resolution"""
        pass
    
    @abstractmethod
    def reserve_files(self, agent_id: str, paths: List[str]) -> bool:
        """File reservation for parallel execution"""
        pass
    
    @abstractmethod
    def release_files(self, agent_id: str) -> bool:
        """Release file reservations"""
        pass
```

### 10.2 Swarm Agent Execution Context

```python
@dataclass
class SwarmContext:
    """Context for swarm agent execution"""
    phase: str
    subphase: str
    task_id: str
    checkpoint_ref: str
    hive_cell_id: str
    file_reservations: Dict[str, List[str]]
    pending_gates: List[str]
    swarm_mail: 'SwarmMail'  # Agent communication channel

class SwarmAgentExecutor:
    """Executes swarm pattern with parallel agents"""
    
    async def execute_subphase(self, 
                                subphase: SubPhase,
                                context: SwarmContext) -> Dict[str, Any]:
        """Execute all tasks in a subphase using swarm agents"""
        
        # Reserve files for parallel agents
        reservations = self._reserve_task_files(subphase.dag)
        context.file_reservations = reservations
        
        # Create agent assignments
        assignments = [
            AgentAssignment(
                agent_id=f"{subphase.agent_role}-{task.id}",
                agent_role=subphase.agent_role,
                task_id=task.id,
                capability=task.capability,
                input=task.input,
                exclusive_paths=self._get_task_paths(task)
            )
            for task in subphase.dag.tasks
        ]
        
        # Execute with coordination
        results = await self.provider.coordinate_agents(assignments)
        
        # Release reservations
        self._release_reservations(reservations)
        
        return self._consolidate_results(results)
```

### 10.3 Swarm Mail Integration

```python
class SwarmMail:
    """Actor-model communication for swarm agents"""
    
    def __init__(self, event_log_path: str):
        self.event_log = event_log_path
        self.agents = set()
        
    def register_agent(self, agent_id: str, role: str):
        """Register an agent in the swarm"""
        self.agents.add(agent_id)
        self._log_event("agent_registered", {
            "agent_id": agent_id,
            "role": role,
            "timestamp": datetime.now().isoformat()
        })
    
    def send(self, from_agent: str, to_agent: str, 
             subject: str, body: Dict) -> str:
        """Send message between agents, persisted to event log"""
        event = {
            "type": "message_sent",
            "from": from_agent,
            "to": to_agent,
            "subject": subject,
            "body": body,
            "timestamp": datetime.now().isoformat()
        }
        return self._log_event("message", event)
    
    def broadcast(self, from_agent: str, subject: str, 
                  body: Dict, roles: List[str] = None) -> List[str]:
        """Broadcast to all agents or specific roles"""
        recipients = self.agents if not roles else self._get_agents_by_roles(roles)
        return [self.send(from_agent, recipient, subject, body) 
                for recipient in recipients]
    
    def reserve(self, agent_id: str, paths: List[str], 
                exclusive: bool = True) -> str:
        """File reservation broadcast"""
        reservation_id = self._log_event("reservation", {
            "agent_id": agent_id,
            "paths": paths,
            "exclusive": exclusive,
            "timestamp": datetime.now().isoformat()
        })
        
        # Notify swarm
        self.broadcast(agent_id, "FILE_RESERVED", {
            "agent_id": agent_id,
            "paths": paths
        })
        
        return reservation_id

# In .spine/events/swarm.log
{"event_id": "ev_001", "type": "agent_registered", "agent_id": "explorer-1", "role": "explorer"}
{"event_id": "ev_002", "type": "message_sent", "from": "planner", "to": "critic", "subject": "PLAN_FOR_REVIEW"}
{"event_id": "ev_003", "type": "reservation", "agent_id": "coder-1", "paths": ["src/auth/**"]}
```

---

## Summary

This provider architecture allows:
- **Runtime flexibility**: Swap providers without code changes
- **Fallback chains**: Primary -> Secondary providers
- **Plugin ecosystem**: Third-party providers via manifest
- **Unified interface**: Same API for all provider types
- **Health monitoring**: Built-in validation and availability checks
- **Swarm agents**: Specialized role-based execution with file reservations