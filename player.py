from chess_tournament.players import Player
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import chess


class TransformerPlayer(Player):
    def __init__(
            self,
            name: str,
            model_id: str = "morinaa/chess-qwen",
            quantization: Optional[str] = "4bit",
            temperature: float = 0.1,
            max_new_tokens: int = 6,
            retries: int = 15
    ):
        super().__init__(name)

        self.model_id = model_id
        self.quantization = quantization
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.retries = retries

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[{self.name}] Loading {self.model_id} on {self.device}")

        # -------------------------
        # Quantization config
        # -------------------------
        quant_config = None

        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        elif quantization is None:
            quant_config = None

        else:
            raise ValueError("quantization must be one of: None, '8bit', '4bit'")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        # Model Loading
        if quant_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto"
            )

        # UCI Regex
        self.uci_re = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b")

    def _build_prompt(self, fen: str) -> str:
        return f"""You are a chess engine.

  Your task is to output the BEST LEGAL MOVE for the given chess position.

  STRICT OUTPUT RULES:
  - Output EXACTLY ONE move
  - UCI format ONLY (examples: e2e4, g1f3, e7e8q)
  - NO explanations, NO punctuation, NO extra text

  FEN: {fen}
  Move:"""
    
    def _extract_move(self, text: str) -> Optional[str]:
        match = self.uci_re.search(text)
        return match.group(0) if match else None
    
    def get_move(self, fen: str) -> Optional[str]:
        prompt = self._build_prompt(fen)
        board = chess.Board(fen)
        legal_ucis = {m.uci() for m in board.legal_moves}

        for attempt in range(1, self.retries + 1):

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            input_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

            move = self._extract_move(decoded)

            if move and move in legal_ucis:
                return random.choice(list(legal_ucis)) if legal_ucis else None

        return None
    

