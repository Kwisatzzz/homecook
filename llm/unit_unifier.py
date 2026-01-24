import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer

from llm.connectors.custom_llama import Llama
from utils.utils import count_tokens, setup_logger

load_dotenv()

_ALLOWED_UNITS = {"g", "ml", "szt"}


class NormalizedOccurrence(BaseModel):
    raw: str = Field(description="Oryginalny opis ilości z wejścia.")
    value: Optional[float] = Field(
        default=None, description="Wartość liczbowa po normalizacji."
    )
    unit: Optional[str] = Field(
        default=None, description="Jednostka po normalizacji: g/ml/szt."
    )
    note: Optional[str] = Field(
        default=None,
        description="Notatka dlaczego nie udało się sparsować lub jakie założenie.",
    )
    confidence: float = Field(
        default=0.7, description="0..1; subiektywna pewność modelu."
    )

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if v not in _ALLOWED_UNITS:
            raise ValueError(f"Invalid unit: {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("confidence must be within [0,1]")
        return v


class IngredientNormalization(BaseModel):
    ingredient: str = Field(description="Nazwa składnika (klucz wejściowy).")
    canonical_unit: Optional[str] = Field(
        default=None, description="Kanoniczna jednostka: g/ml/szt albo null."
    )
    normalized: List[NormalizedOccurrence] = Field(
        description="Lista wystąpień po normalizacji."
    )

    @field_validator("canonical_unit")
    @classmethod
    def validate_canonical_unit(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if v not in _ALLOWED_UNITS:
            raise ValueError(f"Invalid canonical_unit: {v}")
        return v


class UnificationResult(BaseModel):
    items: List[IngredientNormalization] = Field(
        description="Lista normalizacji per składnik."
    )


class UnitUnifier:
    """
    LLM-first normalizer.
    units.json przechowuje WYŁĄCZNIE canonical_unit dla składnika (g/ml/szt lub null).
    """

    def __init__(
        self,
        units_path: str = "data/units.json",
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        safety_buffer_percentage: float = 0.05,
        logger: logging.Logger = None,
        logging_level: str = "INFO",
        endpoint: str = os.environ.get(
            "LLM_ENDPOINT", "http://10.100.1.13:11434/v1/chat/completions"
        ),
        model: str = os.environ.get(
            "LLM_MODEL_NAME", "casperhansen/llama-3.3-70b-instruct-awq"
        ),
    ) -> None:
        self.units_path = Path(units_path)
        self._lock = threading.Lock()

        self.logger = logger or setup_logger(
            "logs/unit_unifier.log", level=logging_level
        )
        self.logger.info("--- Initializing UnitUnifier ---")
        self.logger.info("units_path=%s", str(self.units_path))

        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.safety_buffer_percentage = safety_buffer_percentage

        self.llm = Llama(
            endpoint=endpoint,
            model=model,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        total_model_tokens = self.llm.get_model_context_window() or 2048

        self.parser = PydanticOutputParser(pydantic_object=UnificationResult)
        format_instructions = self.parser.get_format_instructions()

        system_prompt_tokens = count_tokens(self.llm.SYSTEM_PROMPT, self.tokenizer)
        fmt_tokens = count_tokens(format_instructions, self.tokenizer)

        template_text = (
            "Jesteś systemem normalizacji ilości składników w przepisach kuchennych.\n"
            "Dostajesz słownik: nazwa składnika -> lista surowych opisów ilości.\n\n"
            "Twoje cele:\n"
            "1) Dla każdego składnika ustal canonical_unit: 'g' / 'ml' / 'szt'.\n"
            "   - Jeśli składnik jest naprawdę nieustalalny bez zgadywania, canonical_unit ustaw na null.\n"
            "2) Dla KAŻDEGO surowego opisu spróbuj zwrócić value (liczba) + unit (g/ml/szt) w canonical_unit.\n"
            "   - Jeśli opis jest relacyjny lub nieprecyzyjny (np. 'tyle samo', 'do smażenia', 'do smaku', 'trochę', 'szczypta')\n"
            "     ustaw value=null, unit=null i dodaj note.\n\n"
            "Heurystyki kuchenne (stosuj):\n"
            "- 'łyżeczka' ~ 5 ml, 'łyżka' ~ 15 ml, 'szklanka' ~ 250 ml.\n"
            "- ułamki, liczby słowne PL ('dwa', 'pół', 'ćwierć') zamieniaj na liczby.\n"
            "- 'kostka masła' ~ 200 g (dla masła).\n"
            "- produkty liczone (jajko, banan, cebula) zwykle 'szt'; ciecze 'ml'; sypkie/stałe 'g'.\n\n"
            "Trwałość:\n"
            "- Otrzymujesz existing_units_db (ingredient -> canonical_unit lub null). Jeśli składnik istnieje w DB,\n"
            "  trzymaj się tej canonical_unit.\n"
            "- Nie zmieniaj canonical_unit, chyba że poprzednia jest jawnie błędna.\n\n"
            "Zwróć WYŁĄCZNIE JSON zgodny ze schematem.\n\n"
            "ingredients_json:\n{ingredients_json}\n\n"
            "existing_units_db:\n{existing_units_db}\n\n"
            "{format_instructions}"
        )

        instruction_tokens = count_tokens(
            template_text.format(
                ingredients_json="{}", existing_units_db="{}", format_instructions=""
            ),
            self.tokenizer,
        )

        available_tokens = (
            total_model_tokens
            - self.max_output_tokens
            - system_prompt_tokens
            - fmt_tokens
            - instruction_tokens
        )
        buffer_size = int(max(0, available_tokens) * self.safety_buffer_percentage)
        self.max_input_tokens = max(256, available_tokens - buffer_size)

        self.prompt = PromptTemplate(
            template=template_text,
            input_variables=["ingredients_json", "existing_units_db"],
            partial_variables={"format_instructions": format_instructions},
        )
        self.chain = self.prompt | self.llm | self.parser

        self._units_db = self._load_units()
        self.logger.info("--- Initialization completed ---")

    def _load_units(self) -> Dict[str, Optional[str]]:
        with self._lock:
            if not self.units_path.exists():
                self.units_path.parent.mkdir(parents=True, exist_ok=True)
                return {}
            try:
                data = json.loads(self.units_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    return {}
                cleaned: Dict[str, Optional[str]] = {}
                for k, v in data.items():
                    if v is None:
                        cleaned[k] = None
                    else:
                        vv = str(v).strip().lower()
                        cleaned[k] = vv if vv in _ALLOWED_UNITS else None
                return cleaned
            except Exception:
                self.logger.exception("Failed to read units.json; using empty db.")
                return {}

    def _save_units(self) -> None:
        with self._lock:
            self.units_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.units_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self._units_db, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.units_path)

    def _sanitize(self, result: UnificationResult) -> UnificationResult:
        for item in result.items:
            if item.canonical_unit not in _ALLOWED_UNITS:
                item.canonical_unit = None

            for occ in item.normalized:
                if occ.unit not in _ALLOWED_UNITS:
                    occ.unit = None
                if occ.value is not None:
                    try:
                        occ.value = float(occ.value)
                    except Exception:
                        occ.value = None
                        occ.unit = None
                        occ.note = (occ.note or "") + " (value not numeric)"
        return result

    def _update_units_db(self, result: UnificationResult) -> int:
        updated = 0
        for item in result.items:
            key = item.ingredient
            new_unit = item.canonical_unit

            if self._units_db.get(key) != new_unit:
                self._units_db[key] = new_unit
                updated += 1

        if updated:
            self._save_units()
        return updated

    def unify(self, ingredients: Dict[str, List[str]]) -> Dict[str, Any]:
        ingredients_json = json.dumps(ingredients, ensure_ascii=False)
        existing_db_json = json.dumps(self._units_db, ensure_ascii=False)

        tok = count_tokens(ingredients_json, self.tokenizer)
        if tok > self.max_input_tokens:
            self.logger.warning(
                "Input too large (%d tokens). Chunking by ingredients.", tok
            )
            items = list(ingredients.items())
            all_items: List[IngredientNormalization] = []

            chunk: Dict[str, List[str]] = {}
            cur = 0
            for k, v in items:
                piece = json.dumps({k: v}, ensure_ascii=False)
                t = count_tokens(piece, self.tokenizer)
                if chunk and (cur + t) > self.max_input_tokens:
                    resp: UnificationResult = self.chain.invoke(
                        {
                            "ingredients_json": json.dumps(chunk, ensure_ascii=False),
                            "existing_units_db": existing_db_json,
                        }
                    )
                    resp = self._sanitize(resp)
                    all_items.extend(resp.items)
                    chunk = {}
                    cur = 0
                chunk[k] = v
                cur += t

            if chunk:
                resp: UnificationResult = self.chain.invoke(
                    {
                        "ingredients_json": json.dumps(chunk, ensure_ascii=False),
                        "existing_units_db": existing_db_json,
                    }
                )
                resp = self._sanitize(resp)
                all_items.extend(resp.items)

            final = UnificationResult(items=all_items)
        else:
            resp: UnificationResult = self.chain.invoke(
                {
                    "ingredients_json": ingredients_json,
                    "existing_units_db": existing_db_json,
                }
            )
            final = self._sanitize(resp)

        updated = self._update_units_db(final)

        normalized_out: Dict[str, Dict[str, Any]] = {}
        unparsed_out: Dict[str, List[str]] = {}
        reasons_out: Dict[str, List[Dict[str, Any]]] = {}

        for item in final.items:
            ing = item.ingredient
            normalized_out.setdefault(
                ing, {"canonical_unit": item.canonical_unit, "values": []}
            )

            for occ in item.normalized:
                if occ.value is not None and occ.unit is not None:
                    normalized_out[ing]["values"].append(
                        {
                            "raw": occ.raw,
                            "value": occ.value,
                            "unit": occ.unit,
                            "confidence": occ.confidence,
                            "note": occ.note,
                        }
                    )
                else:
                    unparsed_out.setdefault(ing, []).append(occ.raw)
                    reasons_out.setdefault(ing, []).append(
                        {
                            "raw": occ.raw,
                            "note": occ.note,
                            "confidence": occ.confidence,
                        }
                    )

        return {
            "units_path": str(self.units_path),
            "units_db_updated": updated,
            "normalized": normalized_out,
            "unparsed": unparsed_out,
            "reasons": reasons_out,
        }

    def get_units_db(self) -> Dict[str, Optional[str]]:
        return dict(self._units_db)
