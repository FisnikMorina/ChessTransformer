from chess_tournament.players import Player
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import chess
import random


class TransformerPlayer(Player):
    def __init__(
            self,
            name: str,
            model_id: str = "morinaa/chess-transformer",
            temperature: float = 0.1,
            max_new_tokens: int = 6,
    ):
        super().__init__(name)

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[{self.name}] Loading {self.model_id} on {self.device}")

        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        except Exception:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.sep_token = SEP_TOKEN
        self.sep_ids   = self.tokenizer.encode(
            SEP_TOKEN, add_special_tokens=False
        )

        total = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[{name}] Ready! ({total:.0f}M parameters on {self.device})")



    def _compute_logprob(self, prefix: str, uci_move: str) -> float:
        try:
            full_ids   = self.tokenizer.encode(
                prefix + uci_move, return_tensors="pt"
            ).to(self.device)
            prefix_ids = self.tokenizer.encode(
                prefix, return_tensors="pt"
            ).to(self.device)

            prefix_len = prefix_ids.shape[1]
            move_len   = full_ids.shape[1] - prefix_len

            if move_len <= 0:
                return float("-inf")

            with torch.no_grad():
                logits    = self.model(full_ids).logits
                log_probs = torch.nn.functional.log_softmax(
                    logits[0], dim=-1
                )

            score = 0.0
            for i in range(move_len):
                pos      = prefix_len - 1 + i
                token_id = full_ids[0, prefix_len + i]
                score   += log_probs[pos, token_id].item()

            return score / move_len

        except Exception:
            return float("-inf")
            
    def _score_move(self, prefix: str, uci_move: str,
                    board: chess.Board) -> float:
        base  = self._compute_logprob(prefix, uci_move)
        bonus = 0.0

        try:
            move = chess.Move.from_uci(uci_move)

            # Capture bonus
            if board.is_capture(move):
                bonus += 1.0

            board.push(move)

            # Check bonus
            if board.is_check():
                bonus += 0.5

            # Checkmate — finish the game!
            if board.is_checkmate():
                bonus += 10.0

            board.pop()

        except Exception:
            pass

        return base + bonus
                        
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
        board       = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        if not legal_moves:
            return None

        if self.model is None:
            return random.choice(legal_moves)

        prefix = fen + self.sep_token
        scores = {}

        for uci in legal_moves:
            scores[uci] = self._score_move(prefix, uci, board)

        sorted_moves = sorted(scores.items(), key=lambda x: -x[1])
        best_move    = sorted_moves[0][0]

        try:
            board.push(chess.Move.from_uci(best_move))
            if board.is_repetition(2) and len(sorted_moves) > 1:
                board.pop()
                best_move = sorted_moves[1][0]
            else:
                board.pop()
        except Exception:
            pass

        return best_move
    

