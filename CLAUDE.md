# CLAUDE.md — Agentic Pipelines
 
## Stack
- Python 3.11+
- Streamlit (interfaces)
- Anthropic SDK / OpenAI SDK (direto, sem wrappers desnecessários)
- Pydantic para validação e schemas
- httpx para chamadas HTTP async
- python-dotenv para env vars
 
## Arquitetura
 
```
project/
├── agents/              # Um arquivo por agente. Cada agente é uma classe.
│   ├── __init__.py
│   ├── base.py          # BaseAgent com interface comum
│   ├── researcher.py
│   └── writer.py
├── pipelines/           # Orquestração de múltiplos agentes
│   ├── __init__.py
│   └── example_pipeline.py
├── tools/               # Funções que agentes podem chamar
│   ├── __init__.py
│   └── web_search.py
├── models/              # Pydantic models para inputs/outputs
│   ├── __init__.py
│   └── schemas.py
├── utils/               # Helpers genéricos
│   ├── __init__.py
│   ├── llm.py           # Client wrapper unificado para LLM calls
│   └── helpers.py
├── ui/                  # Streamlit pages
│   └── app.py
├── config.py            # Settings centralizados
├── .env
├── .env.example
├── requirements.txt
└── README.md
```
 
## Princípios de código
 
### Legibilidade acima de tudo
- Código é escrito pra ser lido. Se precisa de comentário pra explicar o que faz, o código está ruim.
- Nomes auto-explicativos. `agent_output` e não `ao`. `build_prompt` e não `bp`.
- Funções curtas. Passou de 20 linhas, provavelmente faz mais de uma coisa — quebre.
- Um arquivo, uma responsabilidade. Cresceu demais, separe.
 
### Simplicidade radical
- Resolva o problema da forma mais direta possível.
- Sem abstrações antes da necessidade. Não crie interface genérica pra algo que só tem uma implementação.
- Sem herança profunda. Máximo 2 níveis: `BaseAgent → AgentEspecífico`.
- Sem metaclasses, decorators complexos, mixins, ABCs desnecessários.
- Se uma função resolve, não crie uma classe. Classes existem pra agentes e models.
- Se um `if/elif` resolve, não crie um dict de dispatch nem um strategy pattern.
- Código inteligente é código simples que qualquer pessoa entende em 5 segundos.
 
### Explícito e previsível
- Sem mágica. Nada de `**kwargs` passado adiante sem tipo. Parâmetros explícitos com type hints.
- Type hints em tudo: parâmetros, retornos, atributos de classe.
- Imports explícitos. `from module import ClasseX`, nunca `from module import *`.
- Sem side effects escondidos. Se muda estado, o nome deixa claro (`save_result`, `update_config`).
- Fluxo linear. O leitor lê de cima pra baixo sem pular entre arquivos.
 
### Funções e métodos
- Cada função faz uma coisa. Se tem `and` no nome (`parse_and_save`), são duas funções.
- Parâmetros posicionais pra obrigatórios, keyword pra opcionais. Máximo 4-5 parâmetros — se precisou de mais, crie um Pydantic model.
- Early return sempre. Valide no início, retorne cedo, evite nesting profundo.
- Sem try/except genérico. Capture exceções específicas. `except Exception` só no top-level do pipeline.
 
### Orientação a objetos com propósito
- Classes existem quando há estado + comportamento juntos. Se só tem comportamento, é função. Se só tem estado, é dataclass/Pydantic.
- Todo agente herda de `BaseAgent`. Interface mínima: `run(input) -> output`.
- Agentes são stateless. Estado vive no pipeline, não no agente.
- Cada agente faz UMA coisa. Dois comportamentos = dois agentes.
- Composição sobre herança. Pipeline recebe lista de agentes, não herda de nada.
- `__init__` só atribui. Sem lógica pesada, sem chamadas de API, sem I/O no construtor.
 
### Robustez sem burocracia
- Toda chamada LLM tem retry com backoff exponencial (tenacity ou loop simples).
- Toda chamada LLM tem timeout.
- Output de LLM sempre passa por validação. Nunca confie no formato.
- Parse defensivo: `try/except` no parse, fallback claro, log do erro.
- Logs em cada step do pipeline: qual agente rodou, tempo, sucesso/falha.
 
### Config e secrets
- API keys SEMPRE via environment variables (.env + python-dotenv).
- Nunca hardcode model names, temperaturas ou max_tokens. Centralize em `config.py`.
- Sem números mágicos espalhados. Settings no topo do módulo ou em config.
 
## Padrões de implementação
 
### BaseAgent
```python
class BaseAgent:
    def __init__(self, name: str, model: str, temperature: float = 0):
        self.name = name
        self.model = model
        self.temperature = temperature
 
    def run(self, input_data: dict) -> dict:
        prompt = self._build_prompt(input_data)
        raw = self._call_llm(prompt)
        return self._parse_output(raw)
 
    def _build_prompt(self, input_data: dict) -> str:
        raise NotImplementedError
 
    def _parse_output(self, raw: str) -> dict:
        raise NotImplementedError
 
    def _call_llm(self, prompt: str) -> str:
        # Implementado em utils/llm.py
        raise NotImplementedError
```
 
### Pipeline
```python
class Pipeline:
    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents
 
    def run(self, initial_input: dict) -> dict:
        data = initial_input
        for agent in self.agents:
            data = agent.run(data)
        return data
```
 
### Streamlit UI
- Arquivo principal: `ui/app.py`.
- UI burra: chama o pipeline, mostra resultado. Zero lógica de negócio na UI.
- `st.status()` ou `st.spinner()` pra feedback durante execução.
- Session state só quando necessário.
 
## Regras para o Claude Code
 
1. Ao criar um novo agente, herdar de BaseAgent. Sem exceção.
2. Ao criar um novo pipeline, instanciar agentes e compor na ordem.
3. Nunca instalar dependência sem adicionar no requirements.txt.
4. Nunca criar arquivo fora da estrutura de pastas definida.
5. Sempre criar .env.example quando adicionar nova env var.
6. Commits atômicos: um agente = um commit, um pipeline = um commit.
7. Se algo não cabe na estrutura, pergunte antes de criar pasta nova.
8. Não usar LangChain, CrewAI ou frameworks pesados se a chamada é simples. SDK direto.
9. Não otimizar antes de funcionar. Primeiro roda, depois melhora.
