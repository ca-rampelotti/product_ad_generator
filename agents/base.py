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
        raise NotImplementedError
