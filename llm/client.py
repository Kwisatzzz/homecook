import argparse
import json
import logging
import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from llm.connectors.custom_llama import Llama
from utils.utils import count_tokens, setup_logger, split_into_chunks

load_dotenv()


class IngredientsDict(BaseModel):
    ingredients: Dict[str, List[str]] = Field(
        description="Słownik składników: nazwa składnika -> lista wystąpień/ilości jako stringi."
    )


class IngredientParser:
    def __init__(
        self,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        safety_buffer_percentage: float = 0.05,
        logger: logging.Logger = None,
        logging_level: str = "INFO",
        endpoint: str = os.environ.get(
            "LLM_ENDPOINT", "http://10.100.1.13:11434/v1/chat/completions"
        ),
        model: str = os.environ.get(
            "LLM_MODEL_NAME", "casperhansen/llama-3-70b-instruct-awq"
        ),
    ) -> None:
        self.logger = logger or setup_logger(
            "logs/ingredient_parser.log", level=logging_level
        )
        self.logger.info("--- Initializing IngredientParser ---")

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

        total_model_tokens = self.llm.get_model_context_window()
        if total_model_tokens is None:
            total_model_tokens = 2048
            self.logger.warning(
                "Could not determine model's context window automatically. "
                f"Falling back to default: {total_model_tokens} tokens."
            )

        self.logger.info(f"Model context window tokens: {total_model_tokens}")
        self.logger.info(f"Max output tokens: -{self.max_output_tokens}")

        system_prompt_tokens = count_tokens(self.llm.SYSTEM_PROMPT, self.tokenizer)
        self.logger.info(f"System prompt tokens: -{system_prompt_tokens}")

        self.parser = PydanticOutputParser(pydantic_object=IngredientsDict)
        format_instructions = self.parser.get_format_instructions()
        pydantic_instruction_tokens = count_tokens(format_instructions, self.tokenizer)
        self.logger.info(
            f"Pydantic format instruction tokens: -{pydantic_instruction_tokens}"
        )

        template_text = (
            "Jesteś parserem list składników z przepisów kulinarnych.\n"
            "Dostajesz surowy tekst (często z nagłówkami typu 'Składniki na...', nawiasami, myślnikami, "
            "opisami typu 'opcjonalnie', itp.).\n\n"
            "Twoje zadanie:\n"
            "1) Zwróć WYŁĄCZNIE JSON zgodny ze schematem.\n"
            "2) W polu 'ingredients' zwróć JEDEN słownik (dict):\n"
            "   - klucz: nazwa składnika (string)\n"
            "   - wartość: LISTA stringów, gdzie każdy string to jedna ilość/miara z tekstu\n\n"
            "3) NIGDY nie dodawaj prefiksów sekcji do kluczy (np. 'farsz:', 'ciasto:', 'panierka:' są ZABRONIONE).\n"
            "4) Jeśli ten sam składnik pojawia się wiele razy w tekście (np. w różnych sekcjach),\n"
            "   to:\n"
            "   - użyj TEGO SAMEGO klucza\n"
            "   - dodaj kolejną wartość do listy (append)\n"
            "   - NIE scalaj, NIE sumuj, NIE interpretuj\n\n"
            "5) Nazwy składników normalizuj minimalistycznie:\n"
            "   - małe litery\n"
            "   - usuń śmieci typu '[więcej]'\n"
            "   - usuń same słowa typu 'około', ale zachowaj sens składnika\n"
            "   - NIE dodawaj komentarzy ani kontekstu sekcji\n\n"
            "6) Wartości w listach mają być SUROWE i lokalne:\n"
            "   - zachowuj jednostki i doprecyzowania (np. '1,5 szklanki (375 ml)')\n"
            "   - jeśli ilość jest opisowa, wpisz ją dosłownie (np. 'do smażenia', 'opcjonalnie')\n\n"
            "7) Jeśli w tekście jest linia typu 'przyprawy: ...', rozbij ją na osobne składniki,\n"
            "   ale nadal stosuj te same zasady (dict + listy).\n\n"
            "8) NIE próbuj ujednolicać nazw (np. 'jajko' vs 'jajka'), NIE poprawiaj logiki przepisu.\n"
            "   Twoim zadaniem jest WYŁĄCZNIE ekstrakcja strukturalna.\n\n"
            "Przykład:\n"
            "Wejście: '1 jajko, 3 jajka, olej do smażenia'\n"
            "Wyjście (przykładowe):\n"
            "{{\n"
            '  "ingredients": {{\n'
            '    "jajko": ["1 szt"],\n'
            '    "jajka": ["3 szt"],\n'
            '    "olej": ["do smażenia"]\n'
            "  }}\n"
            "}}\n\n"
            "Tekst:\n{text}\n\n"
            "{format_instructions}"
        )

        instruction_template_text = template_text.format(
            text="", format_instructions=""
        )
        instruction_prompt_tokens = count_tokens(
            instruction_template_text, self.tokenizer
        )
        self.logger.info(f"Instruction prompt tokens: -{instruction_prompt_tokens}")

        available_tokens = (
            total_model_tokens
            - self.max_output_tokens
            - system_prompt_tokens
            - pydantic_instruction_tokens
            - instruction_prompt_tokens
        )

        buffer_size = int(available_tokens * self.safety_buffer_percentage)
        self.max_tokens = available_tokens - buffer_size

        self.logger.info(
            f"Safety buffer tokens ({self.safety_buffer_percentage:.0%}): -{buffer_size}"
        )
        self.logger.info(f"Max input tokens: {self.max_tokens}")

        self.prompt = PromptTemplate(
            template=template_text,
            input_variables=["text"],
            partial_variables={"format_instructions": format_instructions},
        )

        self.chain = self.prompt | self.llm | self.parser
        self.logger.info("--- Initialization completed ---")

    def parse(self, text: str) -> Dict[str, List[str]]:
        tokens_counter = count_tokens(text, self.tokenizer)

        if tokens_counter > self.max_tokens:
            chunks = split_into_chunks(text, self.tokenizer, self.max_tokens)
            merged: Dict[str, str] = {}

            for chunk in chunks:
                response: IngredientsDict = self.chain.invoke({"text": chunk})
                # merge (last write wins on collisions)
                for k, v in response.ingredients.items():
                    merged[k] = v

            output_dict = merged
        else:
            response: IngredientsDict = self.chain.invoke({"text": text})
            output_dict = response.ingredients

        self.logger.info(
            f"IngredientParser:\nOUTPUT: {json.dumps(output_dict, ensure_ascii=False, indent=4)}"
        )
        return output_dict


def _read_input_text(text_arg: str | None, file_arg: str | None) -> str:
    if file_arg:
        with open(file_arg, "r", encoding="utf-8") as f:
            return f.read()
    if text_arg is None:
        raise ValueError("Provide either --text or --file.")
    return text_arg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse ingredients list into a dict: ingredient -> amount."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--text",
        type=str,
        help="Raw ingredients text (wrap in quotes).",
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path to a .txt file with ingredients text.",
    )

    args = parser.parse_args()

    input_text = _read_input_text(args.text, args.file)

    ingredient_parser = IngredientParser()
    result = ingredient_parser.parse(input_text)
