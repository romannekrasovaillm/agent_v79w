import asyncio
import json
import re
import time
import base64
import uuid
import urllib3
import threading
import hashlib
import math
import random
import string
import logging
import requests
import ast
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
from urllib.parse import urlparse
import trafilatura
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Set
from pathlib import Path
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn не установлен. Семантический поиск будет недоступен.")

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK не установлен. Обработка текста будет ограничена.")

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup не установлен. Парсинг HTML будет ограничен.")

try:
    import html2text
    HTML2TEXT_AVAILABLE = True
except ImportError:
    HTML2TEXT_AVAILABLE = False
    logging.warning("html2text не установлен. Конвертация HTML в markdown недоступна.")

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logging.warning("duckduckgo-search не установлен. Веб-поиск будет недоступен.")

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("Playwright не установлен. Браузер будет недоступен.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# CURRENT DATE PLACEHOLDER - Update this with the actual current date
CURRENT_DATE = datetime.now()
CURRENT_DATE_STR = CURRENT_DATE.strftime("%Y-%m-%d")
CURRENT_DATE_FORMATTED = CURRENT_DATE.strftime("%A, %B %d, %Y")

# Download NLTK data if available
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt')
        except:
            NLTK_AVAILABLE = False


@dataclass
class TaskContext:
    """Контекст задачи с информацией о намерениях пользователя."""
    query: str
    intent: str
    requires_search: bool = False
    requires_browser: bool = False
    requires_computation: bool = False
    requires_terminal: bool = False
    complexity: str = "simple"  # simple, medium, complex
    domain: str = "general"
    keywords: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    urgency: str = "normal"  # low, normal, high, critical
    expected_sources: int = 3
    temporal_context: str = "current"  # historical, current, future
    user_goal: str = "information"  # information, action, analysis, export
    confidence_score: float = 0.8
    meta_analysis: Dict[str, Any] = field(default_factory=dict)
    primary_objective: str = ""
    focus_points: List[str] = field(default_factory=list)
    output_expectations: List[str] = field(default_factory=list)
    verification_checks: List[str] = field(default_factory=list)
    prohibited_actions: List[str] = field(default_factory=list)
    focus_summary: str = ""
    

@dataclass
class ExecutionPlan:
    """План выполнения задачи."""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_time: float = 0.0
    confidence: float = 0.8
    fallback_plan: Optional[List[Dict[str, Any]]] = None
    reasoning: str = ""
    risk_assessment: Dict[str, str] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    adaptability_level: str = "medium"  # low, medium, high
    current_step_index: int = 0
    completed_steps: int = 0
    progress_notes: List[str] = field(default_factory=list)


@dataclass
class TodoItem:
    """Представляет задачу для Planning Tool."""
    content: str
    status: str = "pending"

    def to_dict(self) -> Dict[str, str]:
        return {"content": self.content, "status": self.status}


@dataclass
class ToolResult:
    """Результат выполнения инструмента."""
    tool_name: str
    success: bool
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    confidence: float = 0.8


class AdvancedIntentAnalyzer:
    """Продвинутый анализатор намерений пользователя с метарассуждениями."""
    
    def __init__(self, gigachat_client):
        self.client = gigachat_client

    def analyze_with_llm(self, query: str) -> TaskContext:
        """Анализирует запрос с помощью LLM для глубокого понимания намерений."""
        
        analysis_prompt = f"""Я - продвинутый аналитик задач. Мне нужно проанализировать запрос пользователя и понять его истинные намерения.

ТЕКУЩАЯ ДАТА: {CURRENT_DATE_FORMATTED}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{query}"

Я проведу глубокий анализ этого запроса:

1. ОСНОВНЫЕ НАМЕРЕНИЯ:
   - Что пользователь действительно хочет получить?
   - Какие скрытые потребности могут быть за этим запросом?
   - Какова конечная цель пользователя?

2. ВРЕМЕННОЙ КОНТЕКСТ:
   - Нужна ли актуальная информация на сегодняшний день?
   - Запрос касается прошлого, настоящего или будущего?
   - Есть ли временные ограничения?

3. НЕОБХОДИМЫЕ ИНСТРУМЕНТЫ:
   - Нужен ли веб-поиск для получения свежей информации?
   - Требуется ли интерактивная работа с браузером?
   - Нужны ли вычисления или анализ данных?
   - Потребуется ли терминал или работа с файлами?

5. КРИТЕРИИ УСПЕХА:
   - Как я пойму, что задача решена правильно?
   - Какой формат ответа будет наиболее полезен?

Я отвечу в формате JSON:
{{
    "intent": "основное намерение",
    "user_goal": "information/action/analysis/export",
    "requires_search": true/false,
    "requires_browser": true/false,
    "requires_computation": true/false,
    "requires_terminal": true/false,
    "complexity": "simple/medium/complex",
    "domain": "область знаний",
    "urgency": "low/normal/high/critical",
    "temporal_context": "historical/current/future",
    "expected_sources": число_источников,
    "keywords": ["ключевое слово 1", "ключевое слово 2"],
    "success_criteria": ["критерий 1", "критерий 2"],
    "reasoning": "подробное объяснение анализа",
    "confidence_score": 0.0-1.0
}}"""

        try:
            messages = [
                {"role": "system", "content": "Ты - эксперт по анализу пользовательских намерений. Отвечай только в формате JSON."},
                {"role": "user", "content": analysis_prompt}
            ]
            
            response = self.client.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=2048
            )
            
            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']
                
                # Извлекаем JSON из ответа
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                    
                    context = TaskContext(
                        query=query,
                        intent=analysis_data.get('intent', 'general'),
                        user_goal=analysis_data.get('user_goal', 'information'),
                        requires_search=analysis_data.get('requires_search', False),
                        requires_browser=analysis_data.get('requires_browser', False),
                        requires_computation=analysis_data.get('requires_computation', False),
                        requires_terminal=analysis_data.get('requires_terminal', False),
                        complexity=analysis_data.get('complexity', 'simple'),
                        domain=analysis_data.get('domain', 'general'),
                        urgency=analysis_data.get('urgency', 'normal'),
                        temporal_context=analysis_data.get('temporal_context', 'current'),
                        expected_sources=analysis_data.get('expected_sources', 3),
                        keywords=analysis_data.get('keywords', []),
                        confidence_score=analysis_data.get('confidence_score', 0.8),
                        meta_analysis={
                            'success_criteria': analysis_data.get('success_criteria', []),
                            'reasoning': analysis_data.get('reasoning', ''),
                            'llm_analysis': True
                        },
                        timestamp=CURRENT_DATE
                    )

                    return self._augment_context_with_focus(context, analysis_data)
        
        except Exception as e:
            logger.warning(f"LLM анализ не удался, используем базовый: {e}")
        
        # Fallback на базовый анализ
        return self._basic_analysis(query)

    def _basic_analysis(self, query: str) -> TaskContext:
        """Базовый анализ намерений (fallback без эвристик)."""
        context = TaskContext(
            query=query,
            intent="general",
            requires_search=False,
            requires_browser=False,
            requires_computation=False,
            requires_terminal=False,
            complexity="simple",
            domain="general",
            keywords=[],
            timestamp=CURRENT_DATE,
            meta_analysis={'llm_analysis': False}
        )

        return self._augment_context_with_focus(context)

    def _augment_context_with_focus(
        self,
        context: TaskContext,
        analysis_data: Optional[Dict[str, Any]] = None
    ) -> TaskContext:
        """Дополняет контекст задач фокусом на намерениях пользователя."""

        try:
            focus_info = self._derive_focus_from_query(context.query, context, analysis_data)
        except Exception as focus_error:  # noqa: F841
            focus_info = {}

        context.primary_objective = focus_info.get('primary_objective') or context.intent or context.query
        context.focus_points = focus_info.get('focus_points', [])
        context.output_expectations = focus_info.get('output_expectations', [])
        context.verification_checks = focus_info.get('verification_checks', [])
        context.prohibited_actions = focus_info.get('prohibited_actions', [])
        context.focus_summary = focus_info.get('summary', '')

        # Сохраняем данные в метаанализе для дальнейшего использования
        context.meta_analysis.setdefault('focus_points', context.focus_points)
        context.meta_analysis.setdefault('output_expectations', context.output_expectations)
        context.meta_analysis.setdefault('verification_checks', context.verification_checks)
        context.meta_analysis.setdefault('prohibited_actions', context.prohibited_actions)
        if context.focus_summary:
            context.meta_analysis.setdefault('focus_summary', context.focus_summary)

        # Объединяем критерии успеха
        base_success = context.meta_analysis.get('success_criteria', []) or []
        focus_success = focus_info.get('success_criteria', []) or []
        combined_success: List[str] = []
        for criterion in base_success + focus_success:
            if criterion and criterion not in combined_success:
                combined_success.append(criterion)
        if combined_success:
            context.meta_analysis['success_criteria'] = combined_success

        return context

    def _derive_focus_from_query(
        self,
        query: str,
        context: Optional[TaskContext] = None,
        analysis_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Извлекает критические требования и ограничения из пользовательского запроса."""

        normalized = re.sub(r"\s+", " ", query.lower()).strip()

        def add_unique(target: List[str], value: str) -> None:
            if value and value not in target:
                target.append(value)

        focus_points: List[str] = []
        output_expectations: List[str] = []
        verification_checks: List[str] = []
        prohibited_actions: List[str] = [
            "Не выдумывать данные без подтверждения официальными источниками",
            "Не завершать задачу без проверки ключевых критериев пользователя"
        ]
        success_criteria: List[str] = []

        # Попытка определить ключевую метрику
        metric_description = ""
        if any(keyword in normalized for keyword in ["ключев", "ставк"]):
            metric_description = "значение ключевой ставки Банка России"
            add_unique(focus_points, "Найти официальное значение ключевой ставки Банка России")
            add_unique(success_criteria, "Получено значение ключевой ставки из надежного источника")

        # Обработка временных ограничений
        quarter_phrase = ""
        quarter_match = re.search(r"(начал[еаи]|конец|середин[аеы])?\s*(\d)[-\s]*квартал[а-я]*\s+(\d{4})", normalized)
        if quarter_match:
            position = (quarter_match.group(1) or "").strip()
            quarter_number = quarter_match.group(2)
            year = quarter_match.group(3)
            quarter_phrase = f"{position + ' ' if position else ''}{quarter_number}-го квартала {year} года".strip()

        if not quarter_phrase:
            quarter_words_match = re.search(
                r"(начал[еаи]|конец|середин[аеы])?\s*(перв|втор|трет|четверт)[^\s]*\s+квартал[а-я]*\s+(\d{4})",
                normalized
            )
            if quarter_words_match:
                position = (quarter_words_match.group(1) or "").strip()
                quarter_word = quarter_words_match.group(2)
                year = quarter_words_match.group(3)
                mapping = {
                    'перв': '1',
                    'втор': '2',
                    'трет': '3',
                    'четверт': '4'
                }
                quarter_number = mapping.get(quarter_word[:5], mapping.get(quarter_word[:4], ""))
                if not quarter_number:
                    quarter_number = mapping.get(quarter_word[:3], "")
                if quarter_number:
                    quarter_phrase = f"{position + ' ' if position else ''}{quarter_number}-го квартала {year} года".strip()

        if quarter_phrase:
            add_unique(focus_points, f"Убедиться, что данные относятся к {quarter_phrase}")
            add_unique(verification_checks, f"Проверить дату значения ключевой ставки на {quarter_phrase}")
            add_unique(success_criteria, f"Указан период: {quarter_phrase}")

        # Требование использования официального источника
        if any(keyword in normalized for keyword in ["цб", "cbr", "центральн", "офици", "сайт"]):
            add_unique(focus_points, "Открыть официальный сайт Банка России (cbr.ru)")
            add_unique(verification_checks, "Сохранить ссылку на страницу Банка России в результатах")
            add_unique(prohibited_actions, "Не опираться на сторонние сайты без верификации на cbr.ru")

        # Требования к сохранению
        if any(keyword in normalized for keyword in ["excel", "xlsx", "эксель", "таблиц", "сохрани", "сохранить"]):
            add_unique(output_expectations, "Сохранить найденное значение в файл формата Excel (.xlsx)")
            add_unique(success_criteria, "Создан и приложен Excel-файл с результатом")

        # Проверка на необходимость вычислений или дополнительной обработки
        if context and (context.requires_computation or context.user_goal in {"analysis", "export"}):
            add_unique(focus_points, "При необходимости выполнить вычисления или экспорт данных")

        # Общие проверки на точность
        add_unique(verification_checks, "Перепроверить значения перед финальным ответом")

        # Определяем основную цель
        primary_objective = ""
        if metric_description:
            primary_objective = f"Найти {metric_description}"
            if quarter_phrase:
                primary_objective += f" на {quarter_phrase}"
            if output_expectations:
                primary_objective += " и подготовить итоговый Excel-файл"
        elif analysis_data and analysis_data.get('intent') not in (None, '', 'general'):
            primary_objective = analysis_data.get('intent', '').strip()

        if not primary_objective:
            primary_objective = query.strip()

        # Подготавливаем сводку
        summary_parts: List[str] = []
        summary_parts.append(f"Цель: {primary_objective}")
        if focus_points:
            summary_parts.append("Ключевые действия: " + "; ".join(focus_points[:3]))
        if verification_checks:
            summary_parts.append("Проверки: " + "; ".join(verification_checks[:2]))
        if output_expectations:
            summary_parts.append("Итог: " + "; ".join(output_expectations))

        summary = ". ".join(summary_parts)

        # Объединяем критерии успеха с анализом модели
        if analysis_data and isinstance(analysis_data.get('success_criteria'), list):
            for item in analysis_data.get('success_criteria'):
                add_unique(success_criteria, item)

        return {
            'primary_objective': primary_objective,
            'focus_points': focus_points,
            'output_expectations': output_expectations,
            'verification_checks': verification_checks,
            'prohibited_actions': prohibited_actions,
            'success_criteria': success_criteria,
            'summary': summary
        }


class AdvancedTaskPlanner:
    """Продвинутый планировщик задач с метарассуждениями."""
    
    def __init__(self, gigachat_client):
        self.client = gigachat_client
    
    def create_smart_plan(self, context: TaskContext) -> ExecutionPlan:
        """Создает умный план выполнения с использованием LLM."""
        
        planning_prompt = f"""Я - опытный планировщик задач. Мне нужно создать оптимальный план выполнения для следующей задачи:

КОНТЕКСТ ЗАДАЧИ:
- Запрос: "{context.query}"
- Намерение: {context.intent}
- Цель пользователя: {context.user_goal}
- Сложность: {context.complexity}
- Область: {context.domain}
- Срочность: {context.urgency}
- Временной контекст: {context.temporal_context}
- Ключевые слова: {', '.join(context.keywords)}

ДОСТУПНЫЕ УНИВЕРСАЛЬНЫЕ ИНСТРУМЕНТЫ:
1. web_search — поиск в интернете
2. web_parse — извлечение контента со страниц
3. browser_navigate — переход на сайт в браузере
4. browser_extract — извлечение данных из браузера
5. browser_click — взаимодействие с элементами
6. wait_dynamic_content — ожидание динамического контента
7. code_execute — выполнение сценариев и команд в изолированном окружении
8. ls — просмотр содержимого рабочей директории
9. read_file — чтение файлов
10. write_file — создание файлов
11. edit_file — изменение существующих файлов

ТЕКУЩАЯ ДАТА: {CURRENT_DATE_FORMATTED}

Я проанализирую задачу и создам оптимальный план:

1. АНАЛИЗ ЗАДАЧИ:
   - Какие данные нужно получить?
   - Какие инструменты будут наиболее эффективны?
   - В каком порядке их лучше использовать?
   - Какие могут быть проблемы и как их избежать?

2. СТРАТЕГИЯ ВЫПОЛНЕНИЯ:
   - Начинать с самых надежных источников
   - Использовать браузер для динамических сайтов
   - Применять вычисления или терминал для анализа данных
   - Фиксировать результаты в файлах при необходимости

3. ОЦЕНКА РИСКОВ:
   - Что может пойти не так?
   - Какие альтернативы подготовить?

Я отвечу в формате JSON:
{{
    "steps": [
        {{
            "tool": "название_инструмента",
            "priority": число,
            "description": "описание шага",
            "parameters": {{"param": "value"}},
            "expected_outcome": "ожидаемый результат",
            "fallback": "альтернативный инструмент"
        }}
    ],
    "reasoning": "подробное объяснение логики плана",
    "estimated_time": время_в_секундах,
    "confidence": 0.0-1.0,
    "risk_assessment": {{
        "high_risk": "описание высокого риска",
        "medium_risk": "описание среднего риска",
        "mitigation": "стратегии снижения рисков"
    }},
    "success_criteria": ["критерий 1", "критерий 2"],
    "adaptability_level": "low/medium/high"
}}"""

        plan_data: Optional[Dict[str, Any]] = None

        try:
            messages = [
                {"role": "system", "content": "Ты - эксперт по планированию задач. Создавай эффективные планы в формате JSON."},
                {"role": "user", "content": planning_prompt}
            ]

            response = self.client.chat(
                messages=messages,
                temperature=0.2,
                max_tokens=2048
            )

            if response and 'choices' in response:
                content = response['choices'][0]['message']['content']
                plan_data, parse_error = self._parse_plan_response(content)

                if plan_data:
                    adapted_steps = [step.copy() for step in plan_data.get('steps', [])]

                    return ExecutionPlan(
                        steps=adapted_steps,
                        reasoning=plan_data.get('reasoning', ''),
                        estimated_time=plan_data.get('estimated_time', 30.0),
                        confidence=plan_data.get('confidence', 0.8),
                        risk_assessment=plan_data.get('risk_assessment', {}),
                        success_criteria=plan_data.get('success_criteria', []),
                        adaptability_level=plan_data.get('adaptability_level', 'medium'),
                        fallback_plan=self._create_fallback_plan(context),
                        current_step_index=0,
                        completed_steps=0,
                        progress_notes=[]
                    )

                if parse_error:
                    logger.debug(f"Не удалось распарсить ответ планировщика: {parse_error}")

        except Exception as e:
            logger.warning(f"LLM планирование не удалось, используем базовое: {e}")

        # Fallback на базовое планирование
        return self._create_basic_plan(context)

    def _parse_plan_response(self, content: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Пытается преобразовать ответ LLM в структуру плана."""
        if not content:
            return None, "Пустой ответ модели"

        candidates = self._extract_json_candidates(content)
        if not candidates:
            return None, "В ответе отсутствует JSON-блок"

        last_error: Optional[str] = None

        for candidate in candidates:
            variants = [candidate]
            normalized = self._normalize_json_text(candidate)
            if normalized and normalized not in variants:
                variants.append(normalized)

            for variant in variants:
                try:
                    parsed = json.loads(variant)
                    if isinstance(parsed, dict):
                        return parsed, None
                except json.JSONDecodeError as json_error:
                    last_error = f"JSONDecodeError: {json_error}"

                    python_ready = self._prepare_for_python_eval(variant)
                    if python_ready:
                        try:
                            parsed = ast.literal_eval(python_ready)
                            if isinstance(parsed, dict):
                                return parsed, None
                        except (ValueError, SyntaxError) as python_error:
                            last_error = f"{python_error}"

        return None, last_error or "Не удалось преобразовать ответ в JSON"

    def _extract_json_candidates(self, content: str) -> List[str]:
        """Извлекает потенциальные JSON-блоки из ответа модели."""
        candidates: List[str] = []
        if not content:
            return candidates

        code_block_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
        for match in code_block_pattern.finditer(content):
            candidate = match.group(1).strip()
            if candidate:
                candidates.append(candidate)

        if candidates:
            return candidates

        start_index: Optional[int] = None
        depth = 0
        for index, char in enumerate(content):
            if char == '{':
                if depth == 0:
                    start_index = index
                depth += 1
            elif char == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start_index is not None:
                        block = content[start_index:index + 1].strip()
                        if block:
                            candidates.append(block)
                        start_index = None

        if not candidates:
            stripped = content.strip()
            if stripped.startswith('{') and stripped.endswith('}'):
                candidates.append(stripped)

        return candidates

    def _normalize_json_text(self, text: str) -> str:
        """Приводит текст к более валидной JSON-форме."""
        if not text:
            return ''

        sanitized = text.strip()
        sanitized = re.sub(r'//.*?(?=\n|$)', '', sanitized)
        sanitized = re.sub(r'/\*.*?\*/', '', sanitized, flags=re.DOTALL)
        sanitized = re.sub(r',\s*(?=[}\]])', '', sanitized)
        sanitized = self._quote_unquoted_keys(sanitized)

        def replace_single_quotes(match: re.Match) -> str:
            inner = match.group(1)
            inner = inner.replace("\\'", "'")
            inner = inner.replace('"', '\\"')
            return f'"{inner}"'

        sanitized = re.sub(r"(?<!\\)'([^'\\]*(?:\\.[^'\\]*)*)'", replace_single_quotes, sanitized)
        return sanitized

    def _quote_unquoted_keys(self, text: str) -> str:
        """Добавляет кавычки к незаключенным в них ключам JSON."""
        if not text:
            return text

        key_pattern = re.compile(
            r'(?P<prefix>[\{\[,]\s*)(?P<key>[A-Za-zА-Яа-я0-9_\-]+(?:\s+[A-Za-zА-Яа-я0-9_\-]+)*)\s*:'
        )
        text = key_pattern.sub(lambda m: f"{m.group('prefix')}\"{m.group('key')}\":", text)

        start_pattern = re.compile(
            r'^(?P<key>[A-Za-zА-Яа-я0-9_\-]+(?:\s+[A-Za-zА-Яа-я0-9_\-]+)*)\s*:',
            re.MULTILINE
        )
        text = start_pattern.sub(lambda m: f'"{m.group("key")}":', text)

        return text

    def _prepare_for_python_eval(self, text: str) -> str:
        """Преобразует JSON-подобный текст к синтаксису Python для literal_eval."""
        if not text:
            return ''

        normalized = self._normalize_json_text(text)
        if not normalized:
            return normalized

        return self._replace_json_literals(normalized)

    def _replace_json_literals(self, text: str) -> str:
        """Заменяет JSON-литералы на эквиваленты Python вне строк."""
        if not text:
            return ''

        result: List[str] = []
        in_single = False
        in_double = False
        index = 0
        length = len(text)

        while index < length:
            char = text[index]

            if char == '"' and not in_single:
                escaped = index > 0 and text[index - 1] == '\\'
                if not escaped:
                    in_double = not in_double
                result.append(char)
                index += 1
                continue

            if char == "'" and not in_double:
                escaped = index > 0 and text[index - 1] == '\\'
                if not escaped:
                    in_single = not in_single
                result.append(char)
                index += 1
                continue

            if not in_single and not in_double:
                if text.startswith('true', index):
                    result.append('True')
                    index += 4
                    continue
                if text.startswith('false', index):
                    result.append('False')
                    index += 5
                    continue
                if text.startswith('null', index):
                    result.append('None')
                    index += 4
                    continue

            result.append(char)
            index += 1

        return ''.join(result)

    def _create_basic_plan(self, context: TaskContext) -> ExecutionPlan:
        """Создает базовый план (fallback)."""
        steps = [
            {
                'tool': 'web_search',
                'priority': 1,
                'description': 'Собрать исходные материалы через веб-поиск',
                'parameters': {'query': context.query},
                'expected_outcome': 'Список релевантных источников'
            },
            {
                'tool': 'browser_navigate',
                'priority': 2,
                'description': 'Открыть наиболее важный источник в браузере',
                'expected_outcome': 'Загруженная страница с целевым контентом'
            },
            {
                'tool': 'browser_extract',
                'priority': 3,
                'description': 'Извлечь ключевую информацию со страницы',
                'expected_outcome': 'Конспект ключевых фактов'
            },
            {
                'tool': 'write_file',
                'priority': 4,
                'description': 'Зафиксировать промежуточные результаты в заметке',
                'expected_outcome': 'Обновленная заметка с выводами'
            }
        ]

        estimated_time = len(steps) * 5.0

        return ExecutionPlan(
            steps=steps,
            estimated_time=estimated_time,
            confidence=0.7,
            reasoning="Базовый план на основе универсальных инструментов",
            fallback_plan=self._create_fallback_plan(context),
            current_step_index=0,
            completed_steps=0,
            progress_notes=[]
        )

    def _create_fallback_plan(self, context: TaskContext) -> List[Dict]:
        """Создает резервный план."""
        return [
            {'tool': 'web_search', 'priority': 1, 'description': 'Собрать информацию по запросу'},
            {'tool': 'browser_extract', 'priority': 2, 'description': 'Извлечь данные из открытого источника'}
        ]


class GigaChatClient:
    """Клиент для работы с GigaChat API."""

    def __init__(self, client_id: str, client_secret: str, verify_ssl: bool = False, 
                 model: str = "GigaChat-2-Max"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.verify_ssl = verify_ssl
        self.model = model
        self.access_token = None
        self.token_expires_at = 0
        self.base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self._lock = threading.Lock()
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Создает requests сессию с retry стратегией."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get_token(self):
        """Получает access token для GigaChat API."""
        with self._lock:
            if self.access_token and time.time() < self.token_expires_at - 60:
                return self.access_token

            credentials = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            headers = {
                'Authorization': f'Basic {credentials}',
                'RqUID': str(uuid.uuid4()),
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            data = {'scope': 'GIGACHAT_API_CORP'}

            try:
                response = self.session.post(self.auth_url, headers=headers, data=data, 
                                           verify=self.verify_ssl, timeout=20)
                response.raise_for_status()
                token_data = response.json()
                self.access_token = token_data['access_token']
                if 'expires_at' in token_data:
                    self.token_expires_at = int(token_data['expires_at']) // 1000
                else:
                    self.token_expires_at = int(time.time() + int(token_data.get('expires_in', 1800)))
                logger.info("Получен новый GigaChat access token")
                return self.access_token
            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка получения GigaChat токена: {e}")
                raise

    def chat(self, messages: List[Dict], functions: Optional[List[Dict]] = None,
             temperature: float = 0.3, max_tokens: int = 4096,
             function_call: Union[str, Dict] = "auto",
             functions_state_id: Optional[str] = None):
        """Отправляет запрос в GigaChat API."""
        token = self._get_token()
        url = f"{self.base_url}/chat/completions"
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        if functions_state_id is not None:
            data["functions_state_id"] = functions_state_id
        
        if functions:
            data["functions"] = functions
            data["function_call"] = function_call

        try:
            response = self.session.post(url, headers=headers, json=data, 
                                       verify=self.verify_ssl, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка GigaChat запроса: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Содержимое ответа: {e.response.text}")
            raise


class WebSearchTool:
    """Инструмент веб-поиска."""
    
    def __init__(self):
        self.available = DDGS_AVAILABLE
        if not self.available:
            logger.warning("DuckDuckGo поиск недоступен")
    
    def search(self, query: str, max_results: int = 5, region: str = 'wt-wt') -> ToolResult:
        """Выполняет веб-поиск."""
        start_time = time.time()
        
        if not self.available:
            return ToolResult(
                tool_name="web_search",
                success=False,
                data=None,
                error="DuckDuckGo search недоступен. Установите: pip install duckduckgo-search"
            )
        
        try:
            # Добавляем контекст текущей даты для временных запросов
            if any(time_word in query.lower() for time_word in ['сегодня', 'вчера', 'на этой неделе', 'текущий', 'актуальный', 'today', 'yesterday', 'current']):
                query = f"{query} {CURRENT_DATE_STR}"
                logger.info(f"Добавлена текущая дата к запросу: {CURRENT_DATE_STR}")
            
            logger.info(f"Выполняется поиск: '{query}'")
            
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query, 
                    region=region, 
                    safesearch='off', 
                    timelimit='y', 
                    max_results=max_results
                ))
            
            if not results:
                return ToolResult(
                    tool_name="web_search",
                    success=False,
                    data=None,
                    error="Результаты поиска не найдены"
                )
            
            # Форматируем результаты
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_result = {
                    "rank": i,
                    "title": result.get("title", ""),
                    "link": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "relevance_score": 1.0 - (i * 0.1),
                    "search_date": CURRENT_DATE_STR
                }
                formatted_results.append(formatted_result)
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                tool_name="web_search",
                success=True,
                data=formatted_results,
                metadata={
                    'query': query,
                    'results_count': len(formatted_results),
                    'execution_time': execution_time,
                    'source': 'DuckDuckGo',
                    'search_date': CURRENT_DATE_STR
                },
                execution_time=execution_time,
                confidence=0.9 if len(formatted_results) >= 3 else 0.6
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Ошибка веб-поиска: {e}")
            return ToolResult(
                tool_name="web_search",
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )


class WebParsingTool:
    """Инструмент парсинга веб-страниц."""
    
    def __init__(self):
        self.available = trafilatura is not None
        if not self.available:
            logger.warning("Trafilatura недоступен")
    
    def parse(self, url: str, extract_focus: str = None) -> ToolResult:
        """Парсит содержимое веб-страницы."""
        start_time = time.time()
        
        if not self.available:
            return ToolResult(
                tool_name="web_parse",
                success=False,
                data=None,
                error="Trafilatura недоступен. Установите: pip install trafilatura"
            )
        
        try:
            logger.info(f"Парсинг URL: {url}")
            
            # Загружаем страницу
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return ToolResult(
                    tool_name="web_parse",
                    success=False,
                    data=None,
                    error=f"Не удалось загрузить контент с {url}"
                )
            
            # Извлекаем содержимое
            content = trafilatura.extract(
                downloaded, 
                include_links=True, 
                include_tables=True,
                include_comments=False,
                include_formatting=True,
                deduplicate=True
            )
            
            if not content:
                # Пробуем BeautifulSoup как fallback
                if BS4_AVAILABLE:
                    soup = BeautifulSoup(downloaded, 'html.parser')
                    content = soup.get_text(separator='\n', strip=True)
                else:
                    return ToolResult(
                        tool_name="web_parse",
                        success=False,
                        data=None,
                        error="Не удалось извлечь контент"
                    )
            
            # Анализируем контент
            content_analysis = {
                'word_count': len(content.split()),
                'has_tables': '|' in content or 'table' in downloaded.lower(),
                'has_lists': '•' in content or '1.' in content or '- ' in content,
                'estimated_read_time': len(content.split()) / 200,  # minutes
                'parsed_date': CURRENT_DATE_STR
            }
            
            # Если указан фокус извлечения, выделяем релевантные части
            if extract_focus and NLTK_AVAILABLE:
                try:
                    sentences = sent_tokenize(content)
                    relevant_sentences = []
                    focus_words = set(extract_focus.lower().split())
                    
                    for sentence in sentences:
                        sentence_words = set(sentence.lower().split())
                        overlap = len(focus_words & sentence_words)
                        if overlap > 0:
                            relevant_sentences.append((overlap, sentence))
                    
                    if relevant_sentences:
                        relevant_sentences.sort(key=lambda x: x[0], reverse=True)
                        top_sentences = [s[1] for s in relevant_sentences[:5]]
                        focused_content = "\n".join(top_sentences)
                        content = f"[РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ]\n{focused_content}\n\n[ПОЛНЫЙ КОНТЕНТ]\n{content[:1000]}..."
                except:
                    pass  # Если не удалось, используем весь контент
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                tool_name="web_parse",
                success=True,
                data=content,
                metadata={
                    'url': url,
                    'content_length': len(content),
                    'content_analysis': content_analysis,
                    'execution_time': execution_time,
                    'extract_focus': extract_focus
                },
                execution_time=execution_time,
                confidence=0.8
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Ошибка парсинга: {e}")
            return ToolResult(
                tool_name="web_parse",
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )


class BrowserTool:
    """Улучшенный инструмент для работы с браузером."""
    
    def __init__(self):
        self.available = PLAYWRIGHT_AVAILABLE
        self.browser = None
        self.context = None
        self.page = None
        self.current_url = None
        
        if not self.available:
            logger.warning("Playwright недоступен")
    
    def start_session(self, headless: bool = True) -> ToolResult:
        """Запускает сессию браузера."""
        if not self.available:
            return ToolResult(
                tool_name="browser_start",
                success=False,
                data=None,
                error="Playwright недоступен. Установите: pip install playwright && playwright install chromium"
            )
        
        try:
            if self.browser:
                self.close_session()
            
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(
                headless=headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            
            self.context = self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                java_script_enabled=True,
                extra_http_headers={
                    'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8'
                }
            )
            
            self.page = self.context.new_page()
            self.page.set_default_timeout(30000)
            
            logger.info("Браузер запущен")
            return ToolResult(
                tool_name="browser_start",
                success=True,
                data="Браузер успешно запущен",
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Ошибка запуска браузера: {e}")
            return ToolResult(
                tool_name="browser_start",
                success=False,
                data=None,
                error=str(e)
            )
    
    def navigate(self, url: str) -> ToolResult:
        """Переходит на указанный URL с улучшенной обработкой динамического контента."""
        start_time = time.time()
        
        if not self.page:
            start_result = self.start_session()
            if not start_result.success:
                return start_result
        
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            logger.info(f"Переход на: {url}")
            
            response = self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            # Ждем загрузки динамического контента
            try:
                self.page.wait_for_load_state("networkidle", timeout=10000)
            except:
                # Если networkidle не сработал, ждем немного
                time.sleep(3)
            
            # Дополнительное ожидание для JavaScript
            try:
                self.page.wait_for_timeout(2000)
                
                # Пытаемся дождаться появления основного контента
                common_selectors = [
                    'main', 'article', '.content', '#content', 
                    '.main', '[role="main"]', 'body'
                ]
                
                for selector in common_selectors:
                    try:
                        self.page.wait_for_selector(selector, timeout=5000)
                        break
                    except:
                        continue
                        
            except Exception as e:
                logger.debug(f"Дополнительное ожидание не удалось: {e}")
            
            self.current_url = self.page.url
            title = self.page.title() or "Без названия"
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                tool_name="browser_navigate",
                success=True,
                data=f"Успешно перешли на {url}",
                metadata={
                    "url": self.current_url,
                    "title": title,
                    "status": response.status if response else None,
                    "execution_time": execution_time,
                    "navigation_date": CURRENT_DATE_STR
                },
                execution_time=execution_time,
                confidence=0.9
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Ошибка навигации: {e}")
            return ToolResult(
                tool_name="browser_navigate",
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )
    
    def extract_content(self, selector: str = None, wait_for_element: bool = True) -> ToolResult:
        """Извлекает контент со страницы с адаптацией под структуру сайта."""
        start_time = time.time()

        if not self.page:
            return ToolResult(
                tool_name="browser_extract",
                success=False,
                data=None,
                error="Браузер не запущен"
            )

        domain = ""
        if self.current_url:
            try:
                domain = urlparse(self.current_url).netloc.lower()
            except Exception:
                domain = ""

        extraction_metadata = {
            'requested_selector': selector,
            'wait_for_element': wait_for_element,
            'url': self.current_url,
            'domain': domain,
            'attempts': []
        }

        try:
            # Подготавливаем страницу: убираем скрытые элементы и мусор
            self._prepare_page_for_extraction()

            if selector and wait_for_element:
                # Ждем появления элемента, но не прерываем процесс при ошибке
                try:
                    self.page.wait_for_selector(selector, timeout=10000)
                except Exception as wait_error:
                    logger.debug(f"Селектор {selector} не появился за отведенное время: {wait_error}")
                    extraction_metadata['selector_wait_timeout'] = True

            content = None
            used_selector_info: Dict[str, Any] = {}

            if selector:
                selector_candidates = self._prepare_selector_candidates(selector, domain)

                for candidate in selector_candidates:
                    candidate_start = time.time()
                    extracted, attempt_info = self._extract_with_candidate(candidate)
                    attempt_info['duration'] = round(time.time() - candidate_start, 3)
                    extraction_metadata['attempts'].append(attempt_info)

                    if extracted:
                        content = extracted
                        used_selector_info = {
                            'used_selector': attempt_info.get('selector'),
                            'selector_type': attempt_info.get('type'),
                            'matched_elements': attempt_info.get('matches'),
                            'origin': attempt_info.get('origin')
                        }
                        break

            fallback_info = None
            fallback_used = False

            if not content:
                fallback_used = True
                fallback_info = self._extract_main_content(domain)
                content = fallback_info.get('text') or "Контент не найден"

            content = self._clean_extracted_text(content)

            # Собираем структурированные данные для повышения полезности результата
            structured_data = self._extract_structured_data()

            execution_time = time.time() - start_time

            if len(extraction_metadata['attempts']) > 20:
                extraction_metadata['attempts'] = extraction_metadata['attempts'][:20]

            fallback_metadata = None
            if fallback_info:
                fallback_metadata = fallback_info.copy()
                if 'text' in fallback_metadata:
                    fallback_metadata['text_preview'] = fallback_metadata['text'][:200]
                    fallback_metadata.pop('text', None)

            extraction_metadata.update({
                'used_selector': used_selector_info.get('used_selector'),
                'used_selector_type': used_selector_info.get('selector_type'),
                'used_selector_origin': used_selector_info.get('origin'),
                'matched_elements': used_selector_info.get('matched_elements'),
                'fallback_used': fallback_used,
                'fallback_details': fallback_metadata,
                'content_length': len(content) if content else 0
            })

            # Уровень уверенности зависит от того, насколько далеко пришлось отклониться от исходного селектора
            confidence = 0.8
            if used_selector_info.get('origin') and used_selector_info['origin'] not in {
                'user', 'user-prefixed', 'user-text'
            }:
                confidence = 0.7
            if fallback_used:
                confidence = 0.6 if content and content != "Контент не найден" else 0.4

            return ToolResult(
                tool_name="browser_extract",
                success=True,
                data=content,
                metadata={
                    'extraction': extraction_metadata,
                    'execution_time': execution_time,
                    'structured_data': structured_data,
                    'extraction_date': CURRENT_DATE_STR
                },
                execution_time=execution_time,
                confidence=confidence
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Ошибка извлечения контента: {e}")
            return ToolResult(
                tool_name="browser_extract",
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )
    
    def _prepare_page_for_extraction(self) -> None:
        """Удаляет скрытые элементы и подготовляет страницу к извлечению."""
        try:
            self.page.evaluate("""
                () => {
                    const selectors = [
                        '[style*="display: none" i]',
                        '[style*="visibility: hidden" i]',
                        '[hidden]',
                        'script',
                        'style',
                        'noscript',
                        'template'
                    ];
                    const elements = document.querySelectorAll(selectors.join(','));
                    elements.forEach(el => {
                        try {
                            el.remove();
                        } catch (err) {
                            /* ignore */
                        }
                    });
                }
            """)
        except Exception as e:
            logger.debug(f"Не удалось подготовить страницу к извлечению: {e}")

    def _clean_extracted_text(self, text: Optional[str]) -> str:
        """Очищает текст от лишних пробелов и переносов."""
        if not text:
            return ""

        cleaned = text.replace('\r\n', '\n')
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
        return cleaned.strip()

    def _prepare_selector_candidates(self, selector: str, domain: str) -> List[Dict[str, Any]]:
        """Создает набор кандидатов селекторов для повышения устойчивости извлечения."""
        if not selector:
            return []

        raw_candidates = [s.strip() for s in re.split(r'\s*\|\|\s*|\n+', selector) if s.strip()]
        if not raw_candidates:
            raw_candidates = [selector.strip()]

        candidates: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str]] = set()

        def add_candidate(value: str, selector_type: str, origin: str) -> None:
            if not value:
                return
            key = (selector_type, value)
            if key in seen:
                return
            seen.add(key)
            candidates.append({'value': value, 'type': selector_type, 'origin': origin})

        for raw in raw_candidates:
            normalized = raw.strip()
            if not normalized:
                continue

            lower = normalized.lower()
            if lower.startswith('css='):
                add_candidate(normalized[4:].strip() or normalized[4:], 'css', 'user-prefixed')
                continue

            if lower.startswith('xpath='):
                add_candidate(normalized[6:].strip() or normalized[6:], 'xpath', 'user-prefixed')
                continue

            if lower.startswith('text='):
                value = normalized[5:].strip()
                add_candidate(value, 'text', 'user-text')
                if re.fullmatch(r'[\w\-]+', value):
                    add_candidate(f'#{value}', 'css', 'text-as-id')
                    add_candidate(f'.{value}', 'css', 'text-as-class')
                continue

            if lower.startswith('role='):
                role_value = normalized.split('=', 1)[1].strip().strip("'\"")
                add_candidate(f'[role="{role_value}"]', 'css', 'role-attribute')
                continue

            if normalized.startswith('//') or normalized.startswith('('):
                add_candidate(normalized, 'xpath', 'user-xpath')
                continue

            add_candidate(normalized, 'css', 'user')

            if re.fullmatch(r'[\w\-]+', normalized):
                add_candidate(f'#{normalized}', 'css', 'auto-id')
                add_candidate(f'.{normalized}', 'css', 'auto-class')

            if re.fullmatch(r'[\w\s\-\.]+', normalized) and ' ' in normalized:
                add_candidate(normalized, 'text', 'user-text-heuristic')

        for domain_selector in self._get_domain_specific_selectors(domain):
            add_candidate(domain_selector, 'css', 'domain-heuristic')

        common_candidates = [
            'main', 'article', '[role="main"]', '#main', '#content',
            '.content', '.main-content', '.article', '.article-body',
            '.post', '.post-content', '.entry-content', '.news-content'
        ]
        for common_selector in common_candidates:
            add_candidate(common_selector, 'css', 'common-heuristic')

        return candidates[:20]

    def _extract_with_candidate(self, candidate: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
        """Пробует извлечь контент с использованием конкретного кандидата."""
        attempt_info = {
            'selector': candidate.get('value'),
            'type': candidate.get('type'),
            'origin': candidate.get('origin')
        }

        try:
            selector_type = candidate.get('type')
            selector_value = candidate.get('value')

            texts: List[str] = []

            if selector_type == 'css':
                texts = self.page.eval_on_selector_all(
                    selector_value,
                    "elements => elements.map(el => (el.innerText || '').trim()).filter(Boolean)"
                )
            elif selector_type == 'xpath':
                texts = self.page.evaluate(
                    """
                    (expression) => {
                        try {
                            const iterator = document.evaluate(
                                expression,
                                document,
                                null,
                                XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
                                null
                            );
                            const results = [];
                            for (let i = 0; i < iterator.snapshotLength; i++) {
                                const node = iterator.snapshotItem(i);
                                if (!node) continue;
                                const text = (node.innerText || node.textContent || '').trim();
                                if (text) {
                                    results.push(text);
                                }
                            }
                            return results;
                        } catch (err) {
                            return [];
                        }
                    }
                    """,
                    selector_value
                )
            elif selector_type == 'text':
                texts = self.page.evaluate(
                    """
                    (text) => {
                        const needle = text ? text.trim().toLowerCase() : '';
                        if (!needle) {
                            return [];
                        }
                        const nodes = Array.from(
                            document.querySelectorAll('p, div, span, h1, h2, h3, h4, h5, h6, li, a')
                        );
                        const matches = [];
                        for (const node of nodes) {
                            const value = (node.innerText || '').trim();
                            if (!value) continue;
                            if (value.toLowerCase().includes(needle)) {
                                matches.push(value);
                            }
                        }
                        return matches;
                    }
                    """,
                    selector_value
                )
            else:
                attempt_info['error'] = f"Неизвестный тип селектора: {selector_type}"
                return None, attempt_info

            unique_texts: List[str] = []
            seen_texts: Set[str] = set()
            for item in texts or []:
                if not item:
                    continue
                normalized = item.strip()
                if not normalized:
                    continue
                if normalized in seen_texts:
                    continue
                seen_texts.add(normalized)
                unique_texts.append(normalized)

            attempt_info['matches'] = len(unique_texts)

            if not unique_texts:
                return None, attempt_info

            combined = "\n\n".join(unique_texts[:5])
            attempt_info['success'] = True
            return combined, attempt_info

        except Exception as e:
            attempt_info['error'] = str(e)[:200]
            attempt_info.setdefault('matches', 0)
            return None, attempt_info

    def _get_domain_specific_selectors(self, domain: str) -> List[str]:
        """Возвращает список селекторов, характерных для конкретных доменов."""
        if not domain:
            return []

        domain = domain.lower()
        domain_map = {
            'wikipedia.org': ['#mw-content-text', '.mw-parser-output'],
            'medium.com': ['article', '.pw-post-body-paragraph'],
            'habr.com': ['.tm-article-presenter__body', '.article-formatted-body'],
            'vc.ru': ['.content', '.article__body'],
            'ria.ru': ['.article__body', '.article__text'],
            'tass.ru': ['.news-content', '.article__text'],
            'bbc.com': ['main', '.ssrcss-uf6wea-RichTextComponentWrapper'],
            'forbes.ru': ['.article__content', '.c-article__body'],
            'rbc.ru': ['.article__text', '.js-article__content'],
            'lenta.ru': ['.topic-body__content', '.js-topic__content'],
            'theguardian.com': ['main', '.article-body-commercial-selector']
        }

        selectors: List[str] = []
        for domain_key, domain_selectors in domain_map.items():
            if domain_key in domain:
                selectors.extend(domain_selectors)

        return selectors

    def _extract_main_content(self, domain: str) -> Dict[str, Any]:
        """Извлекает основной контент страницы с учетом общих и доменных шаблонов."""
        candidate_selectors = self._get_domain_specific_selectors(domain)
        common_candidates = [
            'main', 'article', '[role="main"]', '#main', '#content',
            '.content', '.main-content', '.article', '.article-body',
            '.post', '.post-content', '.entry-content', '.news-content'
        ]

        for selector in common_candidates:
            if selector not in candidate_selectors:
                candidate_selectors.append(selector)

        fallback_result = {}

        try:
            fallback_result = self.page.evaluate(
                r"""
                (candidates) => {
                    if (!Array.isArray(candidates)) {
                        candidates = [];
                    }

                    for (const candidate of candidates) {
                        try {
                            const element = document.querySelector(candidate);
                            if (!element) continue;
                            const style = window.getComputedStyle(element);
                            if (style && (style.display === 'none' || style.visibility === 'hidden')) {
                                continue;
                            }
                            const text = (element.innerText || '').trim();
                            if (text && text.length > 160) {
                                return {
                                    text,
                                    selector: candidate,
                                    strategy: 'candidate_selector'
                                };
                            }
                        } catch (err) {
                            continue;
                        }
                    }

                    const blocks = Array.from(document.querySelectorAll('main, article, section, div'));
                    let best = null;

                    for (const element of blocks) {
                        if (!element) continue;
                        const style = window.getComputedStyle(element);
                        if (style && (style.display === 'none' || style.visibility === 'hidden')) {
                            continue;
                        }
                        const text = (element.innerText || '').trim();
                        if (!text || text.length < 200) {
                            continue;
                        }

                        if (!best || text.length > best.text.length) {
                            let descriptor = element.tagName.toLowerCase();
                            if (element.id) {
                                descriptor += '#' + element.id;
                            } else if (element.className) {
                                const className = element.className.toString().trim().split(/\s+/).filter(Boolean).slice(0, 2).join('.');
                                if (className) {
                                    descriptor += '.' + className;
                                }
                            }
                            best = {
                                text,
                                selector: descriptor,
                                strategy: 'largest_visible_block'
                            };
                        }
                    }

                    if (best) {
                        return best;
                    }

                    const bodyText = (document.body ? document.body.innerText : '').trim();
                    return {
                        text: bodyText,
                        selector: 'body',
                        strategy: 'full_body'
                    };
                }
                """,
                candidate_selectors
            ) or {}
        except Exception as e:
            logger.debug(f"Не удалось извлечь основной контент по кандидатам: {e}")

        text = fallback_result.get('text', '') if isinstance(fallback_result, dict) else ''
        selector_used = fallback_result.get('selector') if isinstance(fallback_result, dict) else None
        strategy = fallback_result.get('strategy') if isinstance(fallback_result, dict) else None

        if not text:
            try:
                text = self.page.evaluate("() => document.body ? document.body.innerText : ''")
            except Exception:
                text = self.page.text_content('body') or ''

            text = text or ''
            if not selector_used:
                selector_used = 'body'
            if not strategy:
                strategy = 'full_body'

        return {
            'text': text,
            'selector': selector_used,
            'strategy': strategy,
            'candidate_pool_size': len(candidate_selectors),
            'candidate_preview': candidate_selectors[:10]
        }

    def _extract_structured_data(self) -> Dict[str, Any]:
        """Извлекает структурированные данные со страницы."""
        try:
            # Извлекаем таблицы
            tables = self.page.evaluate("""
                () => {
                    const tables = Array.from(document.querySelectorAll('table'));
                    return tables.map(table => {
                        const rows = Array.from(table.querySelectorAll('tr'));
                        return rows.map(row => 
                            Array.from(row.querySelectorAll('td, th')).map(cell => cell.innerText.trim())
                        );
                    });
                }
            """)
            
            # Извлекаем списки
            lists = self.page.evaluate("""
                () => {
                    const lists = Array.from(document.querySelectorAll('ul, ol'));
                    return lists.map(list => 
                        Array.from(list.querySelectorAll('li')).map(item => item.innerText.trim())
                    );
                }
            """)
            
            return {
                'tables': tables,
                'lists': lists,
                'has_tables': len(tables) > 0,
                'has_lists': len(lists) > 0
            }
        except:
            return {}
    
    def wait_for_dynamic_content(self, timeout: int = 10000) -> ToolResult:
        """Ждет загрузки динамического контента."""
        try:
            # Комбинация методов ожидания
            strategies = [
                lambda: self.page.wait_for_load_state("networkidle", timeout=timeout),
                lambda: self.page.wait_for_timeout(3000),
                lambda: self.page.wait_for_function("() => document.readyState === 'complete'", timeout=timeout)
            ]
            
            for strategy in strategies:
                try:
                    strategy()
                    break
                except:
                    continue
            
            return ToolResult(
                tool_name="wait_dynamic_content",
                success=True,
                data="Динамический контент загружен",
                confidence=0.8
            )
            
        except Exception as e:
            return ToolResult(
                tool_name="wait_dynamic_content",
                success=False,
                data=None,
                error=str(e)
            )
    
    def click_element(self, selector: str) -> ToolResult:
        """Кликает по элементу."""
        start_time = time.time()
        
        if not self.page:
            return ToolResult(
                tool_name="browser_click",
                success=False,
                data=None,
                error="Браузер не запущен"
            )
        
        try:
            logger.info(f"Клик по элементу: {selector}")
            
            # Пробуем разные стратегии поиска элемента
            strategies = [
                lambda: self.page.click(selector, timeout=5000),
                lambda: self.page.click(f"#{selector}", timeout=5000),
                lambda: self.page.click(f".{selector}", timeout=5000),
                lambda: self.page.click(f"text={selector}", timeout=5000),
            ]
            
            success = False
            for strategy in strategies:
                try:
                    strategy()
                    success = True
                    break
                except:
                    continue
            
            if not success:
                return ToolResult(
                    tool_name="browser_click",
                    success=False,
                    data=None,
                    error=f"Не удалось найти элемент: {selector}"
                )
            
            # Ждем изменений на странице
            self.page.wait_for_load_state("networkidle", timeout=3000)
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                tool_name="browser_click",
                success=True,
                data=f"Успешно кликнули по {selector}",
                metadata={
                    'selector': selector,
                    'execution_time': execution_time,
                    'new_url': self.page.url
                },
                execution_time=execution_time,
                confidence=0.8
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Ошибка клика: {e}")
            return ToolResult(
                tool_name="browser_click",
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )
    
    def close_session(self) -> ToolResult:
        """Закрывает сессию браузера."""
        try:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if hasattr(self, 'playwright'):
                self.playwright.stop()
            
            self.page = None
            self.context = None
            self.browser = None
            self.current_url = None
            
            logger.info("Браузер закрыт")
            return ToolResult(
                tool_name="browser_close",
                success=True,
                data="Браузер закрыт",
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Ошибка закрытия браузера: {e}")
            return ToolResult(
                tool_name="browser_close",
                success=False,
                data=None,
                error=str(e)
            )


class CodeExecutor:
    """Исполнитель Python кода в изолированном окружении."""
    
    def __init__(self):
        self.globals = {
            '__builtins__': __builtins__,
            'json': json,
            'math': math,
            'random': random,
            're': re,
            'time': time,
            'datetime': datetime,
            'CURRENT_DATE': CURRENT_DATE,
            'CURRENT_DATE_STR': CURRENT_DATE_STR
        }
        
        # Добавляем numpy если доступен
        if SKLEARN_AVAILABLE:
            import numpy as np
            self.globals['np'] = np
        
    def execute(self, code: str) -> ToolResult:
        """Выполняет Python код."""
        start_time = time.time()
        
        try:
            # Создаем изолированное пространство имен
            local_namespace = {}
            
            # Перехватываем вывод
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, self.globals, local_namespace)
            
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()
            
            # Извлекаем результат
            result_data = local_namespace.get('result', stdout if stdout else "Код выполнен успешно")
            
            execution_time = time.time() - start_time

            return ToolResult(
                tool_name="code_execute",
                success=True,
                data=result_data,
                metadata={
                    'execution_time': execution_time,
                    'code_length': len(code),
                    'output': stdout,
                    'errors': stderr,
                    'execution_date': CURRENT_DATE_STR
                },
                execution_time=execution_time,
                confidence=0.8
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Ошибка выполнения кода: {e}")
            
            return ToolResult(
                tool_name="code_execute",
                success=False,
                data=None,
                error=str(e),
                metadata={
                    'execution_time': execution_time,
                    'code_length': len(code)
                },
                execution_time=execution_time
            )
    

class FileSystemTools:
    """Реальные инструменты работы с файловой системой для метапамяти агента."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None, encoding: str = "utf-8"):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "agent_workspace"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.encoding = encoding

    def _resolve_path(self, file_path: Optional[Union[str, Path]]) -> Path:
        if file_path in (None, "", "."):
            return self.base_dir

        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path

        resolved = path.resolve()
        base_resolved = self.base_dir.resolve()
        if not resolved.is_relative_to(base_resolved):
            raise ValueError("Доступ к файлам вне рабочей директории запрещен")

        return resolved

    def _relative_path(self, path: Path) -> str:
        try:
            relative = path.relative_to(self.base_dir)
            return "." if str(relative) == "." else str(relative)
        except ValueError:
            return str(path)

    def _describe_path(self, path: Path) -> Dict[str, Any]:
        info = {
            "name": path.name,
            "path": self._relative_path(path),
            "type": "directory" if path.is_dir() else "file"
        }
        try:
            stats = path.stat()
            info["size"] = stats.st_size
            info["modified"] = datetime.fromtimestamp(stats.st_mtime).isoformat()
        except OSError:
            info["size"] = None
            info["modified"] = None
        return info

    def list_files(self, path: Optional[str] = None, recursive: bool = False,
                   include_hidden: bool = False) -> ToolResult:
        """Возвращает список файлов и директорий в рабочем пространстве."""
        try:
            target = self._resolve_path(path)

            if target.exists() and target.is_file():
                description = self._describe_path(target)
                return ToolResult(
                    tool_name="ls",
                    success=True,
                    data=[description["path"]],
                    metadata={
                        "base_directory": str(self.base_dir),
                        "entries": [description],
                        "target": description["path"],
                        "recursive": False
                    }
                )

            if not target.exists():
                return ToolResult(
                    tool_name="ls",
                    success=False,
                    data=None,
                    error="Указанный каталог не найден"
                )

            if not target.is_dir():
                return ToolResult(
                    tool_name="ls",
                    success=False,
                    data=None,
                    error="Путь не является директорией"
                )

            if recursive:
                entries_iter = target.rglob("*")
            else:
                entries_iter = target.iterdir()

            entries = []
            simple_listing = []
            for entry in sorted(entries_iter, key=lambda p: str(self._relative_path(p)).lower()):
                relative_path = self._relative_path(entry)
                if not include_hidden:
                    parts = Path(relative_path).parts if relative_path not in ("", ".") else ()
                    if any(part.startswith('.') for part in parts):
                        continue
                description = self._describe_path(entry)
                entries.append(description)
                simple_listing.append(description["path"])

            return ToolResult(
                tool_name="ls",
                success=True,
                data=simple_listing,
                metadata={
                    "base_directory": str(self.base_dir),
                    "entries": entries,
                    "target": self._relative_path(target),
                    "recursive": recursive
                }
            )

        except Exception as e:
            return ToolResult(
                tool_name="ls",
                success=False,
                data=None,
                error=str(e)
            )

    def read_file(self, file_path: str, offset: int = 0, limit: int = 2000) -> ToolResult:
        """Читает файл и возвращает его содержимое с номерами строк."""
        try:
            path = self._resolve_path(file_path)
            if not path.exists() or not path.is_file():
                return ToolResult(
                    tool_name="read_file",
                    success=False,
                    data=None,
                    error="Файл не найден"
                )

            with path.open("r", encoding=self.encoding) as f:
                lines = f.readlines()

            total_lines = len(lines)
            start = max(offset, 0)
            end = total_lines if limit is None else min(total_lines, start + max(limit, 0))

            sliced_lines = lines[start:end]
            if not sliced_lines and total_lines == 0:
                content = "Файл пуст"
            else:
                numbered = []
                for idx, line in enumerate(sliced_lines, start=start + 1):
                    stripped_line = line.rstrip("\n")
                    numbered.append(f"{idx:>6}⟶{stripped_line}")
                content = "\n".join(numbered)

            return ToolResult(
                tool_name="read_file",
                success=True,
                data=content,
                metadata={
                    "path": self._relative_path(path),
                    "total_lines": total_lines,
                    "offset": start,
                    "limit": limit,
                    "lines_returned": len(sliced_lines)
                }
            )

        except Exception as e:
            return ToolResult(
                tool_name="read_file",
                success=False,
                data=None,
                error=str(e)
            )

    def write_file(self, file_path: str, content: str, overwrite: bool = False) -> ToolResult:
        """Создает или перезаписывает файл."""
        try:
            path = self._resolve_path(file_path)
            if path.exists() and not overwrite:
                return ToolResult(
                    tool_name="write_file",
                    success=False,
                    data=None,
                    error="Файл уже существует. Укажите overwrite=True для перезаписи"
                )

            path.parent.mkdir(parents=True, exist_ok=True)
            text_content = "" if content is None else str(content)
            path.write_text(text_content, encoding=self.encoding)

            return ToolResult(
                tool_name="write_file",
                success=True,
                data=f"Файл сохранен: {self._relative_path(path)}",
                metadata={
                    "path": self._relative_path(path),
                    "bytes_written": len(text_content.encode(self.encoding)),
                    "overwrite": overwrite
                }
            )

        except Exception as e:
            return ToolResult(
                tool_name="write_file",
                success=False,
                data=None,
                error=str(e)
            )

    def edit_file(self, file_path: str, old_string: str, new_string: str,
                  replace_all: bool = False) -> ToolResult:
        """Изменяет существующий файл, заменяя указанную строку."""
        try:
            if not old_string:
                return ToolResult(
                    tool_name="edit_file",
                    success=False,
                    data=None,
                    error="Параметр old_string не может быть пустым"
                )

            path = self._resolve_path(file_path)
            if not path.exists() or not path.is_file():
                return ToolResult(
                    tool_name="edit_file",
                    success=False,
                    data=None,
                    error="Файл не найден"
                )

            text = path.read_text(encoding=self.encoding)
            occurrences = text.count(old_string)

            if occurrences == 0:
                return ToolResult(
                    tool_name="edit_file",
                    success=False,
                    data=None,
                    error="Строка для замены не найдена"
                )

            if occurrences > 1 and not replace_all:
                return ToolResult(
                    tool_name="edit_file",
                    success=False,
                    data=None,
                    error="Строка встречается несколько раз. Уточните контекст или используйте replace_all=True"
                )

            if replace_all:
                new_text = text.replace(old_string, new_string)
                replacements = occurrences
            else:
                new_text = text.replace(old_string, new_string, 1)
                replacements = 1

            path.write_text(new_text, encoding=self.encoding)

            return ToolResult(
                tool_name="edit_file",
                success=True,
                data=f"Заменено {replacements} фрагмент(ов) в {self._relative_path(path)}",
                metadata={
                    "path": self._relative_path(path),
                    "replacements": replacements,
                    "replace_all": replace_all
                }
            )

        except Exception as e:
            return ToolResult(
                tool_name="edit_file",
                success=False,
                data=None,
                error=str(e)
            )

    def ensure_directory(self, directory: str) -> Path:
        """Гарантирует существование директории в рабочем пространстве."""
        path = self._resolve_path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_path(self, file_path: str) -> Path:
        """Публичная обертка для получения абсолютного пути внутри рабочей директории."""
        return self._resolve_path(file_path)


class MetacognitionManager:
    """Управляет метапамятью агента через файловую систему."""

    def __init__(self, fs_tools: FileSystemTools):
        self.fs_tools = fs_tools
        self.session_dir: Optional[str] = None

    def _safe_session_name(self, query: str) -> str:
        sanitized = re.sub(r"[^\wа-яА-ЯёЁ-]+", "_", query, flags=re.UNICODE)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        if not sanitized:
            sanitized = "task"
        return sanitized[:60]

    def start_session(self, query: str, context: TaskContext, plan: ExecutionPlan) -> Optional[str]:
        """Создает файловую структуру для хранения метаданных текущей задачи."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = f"metacognition/{timestamp}_{self._safe_session_name(query)}"
            self.fs_tools.ensure_directory(session_dir)
            self.session_dir = session_dir

            metadata = {
                "query": query,
                "created_at": datetime.now().isoformat(),
                "workspace": str(self.fs_tools.base_dir.resolve()),
                "session_directory": session_dir
            }

            context_result = self.fs_tools.write_file(
                f"{session_dir}/context.json",
                json.dumps(asdict(context), ensure_ascii=False, indent=2, default=str),
                overwrite=True
            )
            if not context_result.success:
                logger.warning(f"Не удалось сохранить контекст: {context_result.error}")

            plan_data = {
                "steps": plan.steps,
                "estimated_time": plan.estimated_time,
                "confidence": plan.confidence,
                "fallback_plan": plan.fallback_plan,
                "reasoning": plan.reasoning,
                "risk_assessment": plan.risk_assessment,
                "success_criteria": plan.success_criteria
            }

            plan_result = self.fs_tools.write_file(
                f"{session_dir}/plan.json",
                json.dumps(plan_data, ensure_ascii=False, indent=2, default=str),
                overwrite=True
            )
            if not plan_result.success:
                logger.warning(f"Не удалось сохранить план: {plan_result.error}")

            plan_lines = [
                "# План выполнения",
                "",
                f"Создан: {datetime.now().isoformat()}",
                "",
            ]
            for idx, step in enumerate(plan.steps, 1):
                description = step.get("description") or step.get("tool", "Шаг без описания")
                priority = step.get("priority")
                note = f" (приоритет: {priority})" if priority is not None else ""
                plan_lines.append(f"{idx}. **{step.get('tool', 'tool')}** — {description}{note}")
            if not plan.steps:
                plan_lines.append("План пуст")

            plan_md_result = self.fs_tools.write_file(
                f"{session_dir}/plan.md",
                "\n".join(plan_lines),
                overwrite=True
            )
            if not plan_md_result.success:
                logger.warning(f"Не удалось сохранить план в Markdown: {plan_md_result.error}")

            metadata_result = self.fs_tools.write_file(
                f"{session_dir}/metadata.json",
                json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
                overwrite=True
            )
            if not metadata_result.success:
                logger.warning(f"Не удалось сохранить метаданные: {metadata_result.error}")

            log_init = self.fs_tools.write_file(
                f"{session_dir}/execution_log.jsonl",
                "",
                overwrite=True
            )
            if not log_init.success:
                logger.warning(f"Не удалось создать файл журнала: {log_init.error}")

            return session_dir

        except Exception as e:
            logger.warning(f"Не удалось инициализировать файловую сессию: {e}")
            self.session_dir = None
            return None

    def record_system_prompt(self, prompt: str) -> None:
        if not self.session_dir:
            return
        try:
            self.fs_tools.write_file(
                f"{self.session_dir}/system_prompt.txt",
                prompt,
                overwrite=True
            )
        except Exception as e:
            logger.warning(f"Не удалось сохранить системный промпт: {e}")

    def update_progress(self, progress: Dict[str, Any]) -> None:
        if not self.session_dir:
            return
        try:
            self.fs_tools.write_file(
                f"{self.session_dir}/progress.json",
                json.dumps(progress, ensure_ascii=False, indent=2, default=str),
                overwrite=True
            )
        except Exception as e:
            logger.warning(f"Не удалось обновить прогресс: {e}")

    def log_tool_execution(self, tool_name: str, arguments: Dict[str, Any], result: ToolResult,
                           plan_progress: Dict[str, Any], note: Optional[str] = None) -> None:
        if not self.session_dir:
            return
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "arguments": arguments,
                "success": result.success,
                "error": result.error,
                "data_preview": str(result.data)[:500] if result.data else None,
                "metadata": result.metadata,
                "confidence": result.confidence,
                "execution_time": result.execution_time,
                "note": note
            }

            log_path = self.fs_tools.resolve_path(f"{self.session_dir}/execution_log.jsonl")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding=self.fs_tools.encoding) as log_file:
                log_file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

            self.update_progress(plan_progress)

        except Exception as e:
            logger.warning(f"Не удалось записать шаг в журнал: {e}")

    def finalize(self, final_answer: Optional[str], summary: Dict[str, Any]) -> None:
        if not self.session_dir:
            return
        try:
            summary_payload = summary.copy()
            summary_payload["finalized_at"] = datetime.now().isoformat()
            self.fs_tools.write_file(
                f"{self.session_dir}/final_summary.json",
                json.dumps(summary_payload, ensure_ascii=False, indent=2, default=str),
                overwrite=True
            )

            if final_answer is not None:
                self.fs_tools.write_file(
                    f"{self.session_dir}/final_answer.md",
                    final_answer,
                    overwrite=True
                )

        except Exception as e:
            logger.warning(f"Не удалось сохранить финальную информацию: {e}")


class PlanningToolManager:
    """Интеграция с Planning Tool (write_todos) для управления сложными задачами."""

    VALID_STATUSES: Set[str] = {"pending", "in_progress", "completed"}

    def __init__(self, metacognition: Optional[MetacognitionManager] = None):
        self.metacognition = metacognition
        self.reset()

    def reset(self) -> None:
        """Сбрасывает состояние планировщика."""
        self.todos: List[TodoItem] = []
        self.active: bool = False
        self.last_state_hash: Optional[str] = None
        self.last_updated_at: Optional[str] = None

    def should_activate(self, context: TaskContext, plan: ExecutionPlan) -> bool:
        """Определяет, нужно ли автоматически включать Planning Tool."""
        if not plan or not getattr(plan, "steps", None):
            return False

        step_count = len(plan.steps)
        complexity = (context.complexity or "simple").lower() if context else "simple"
        requires_multitool = any([
            step_count >= 3,
            complexity in {"medium", "complex"},
            context.requires_browser if context else False,
            context.requires_search if context else False,
            context.requires_computation if context else False,
            context.requires_terminal if context else False,
            context.user_goal in {"analysis", "action", "export"} if context else False
        ])

        return requires_multitool

    def initialize_from_plan(self, plan: ExecutionPlan) -> List[Dict[str, str]]:
        """Создает todo-список на основе плана выполнения."""
        self.todos = []

        if not plan or not getattr(plan, "steps", None):
            self.reset()
            return []

        for idx, step in enumerate(plan.steps, 1):
            description = step.get("description") or step.get("expected_outcome") or step.get("tool", "Шаг")
            content = f"Шаг {idx}: {description}".strip()
            self.todos.append(TodoItem(content=content, status="pending"))

        if self.todos:
            self.todos[0].status = "in_progress"

        self.active = bool(self.todos)
        self._record_state_change()
        return self.to_dicts()

    def _serialize_state(self) -> str:
        return json.dumps([todo.to_dict() for todo in self.todos], ensure_ascii=False, sort_keys=True)

    def _record_state_change(self) -> None:
        self.last_state_hash = self._serialize_state() if self.todos else None
        self.last_updated_at = datetime.now().isoformat()

    def _enforce_progress_rules(self) -> bool:
        """Поддерживает корректные статусы задач согласно правилам инструмента."""
        changed = False

        for todo in self.todos:
            if todo.status not in self.VALID_STATUSES:
                todo.status = "pending"
                changed = True

        if not self.todos:
            return changed

        has_active = any(todo.status == "in_progress" for todo in self.todos if todo.status != "completed")
        if not has_active:
            for todo in self.todos:
                if todo.status == "pending":
                    todo.status = "in_progress"
                    changed = True
                    break

        return changed

    def to_dicts(self) -> List[Dict[str, str]]:
        return [todo.to_dict() for todo in self.todos]

    def get_state(self) -> Dict[str, Any]:
        """Возвращает текущее состояние Planning Tool."""
        return {
            "active": self.active,
            "todos": self.to_dicts(),
            "last_updated_at": self.last_updated_at
        }

    def handle_write_todos(self, todos_payload: Any) -> ToolResult:
        """Обрабатывает вызов функции write_todos."""
        start_time = time.time()

        if todos_payload is None:
            todos_payload = []

        if not isinstance(todos_payload, list):
            return ToolResult(
                tool_name="write_todos",
                success=False,
                data={"errors": ["Параметр 'todos' должен быть списком"]},
                error="Некорректный формат параметров",
                execution_time=time.time() - start_time,
                confidence=0.0
            )

        normalized: List[TodoItem] = []
        errors: List[str] = []

        for index, item in enumerate(todos_payload, 1):
            if not isinstance(item, dict):
                errors.append(f"Элемент #{index} не является объектом")
                continue

            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()

            if not content:
                errors.append(f"Элемент #{index} содержит пустое описание")
                continue

            if status not in self.VALID_STATUSES:
                errors.append(f"Элемент #{index} имеет недопустимый статус '{status}'")
                status = "pending"

            normalized.append(TodoItem(content=content, status=status))

        if not normalized:
            return ToolResult(
                tool_name="write_todos",
                success=False,
                data={"errors": errors or ["Не удалось сформировать список задач"]},
                error="Список задач пуст или некорректен",
                execution_time=time.time() - start_time,
                confidence=0.0
            )

        self.todos = normalized
        self.active = True
        auto_adjusted = self._enforce_progress_rules()
        self._record_state_change()

        metadata = {
            "count": len(self.todos),
            "active": self.active,
            "auto_adjusted": auto_adjusted,
        }
        if errors:
            metadata["warnings"] = errors

        result_data = {
            "todos": self.to_dicts(),
            "active": self.active,
            "last_updated_at": self.last_updated_at
        }

        return ToolResult(
            tool_name="write_todos",
            success=True,
            data=result_data,
            metadata=metadata,
            error=None,
            execution_time=time.time() - start_time,
            confidence=1.0 if self.active else 0.8
        )

    def sync_with_plan(self, plan: ExecutionPlan) -> bool:
        """Синхронизирует статусы todo со статусами шагов плана."""
        if not self.active or not self.todos or not plan or not getattr(plan, "steps", None):
            return False

        changed = False

        for step in plan.steps:
            order = step.get("order")
            if order is None or order >= len(self.todos):
                continue

            plan_status = step.get("status")

            if plan_status in {"completed", "skipped"}:
                desired_status = "completed"
            elif plan_status == "in_progress":
                desired_status = "in_progress"
            else:
                desired_status = "pending"

            if self.todos[order].status != desired_status:
                self.todos[order].status = desired_status
                changed = True

        if self._enforce_progress_rules():
            changed = True

        serialized_state = self._serialize_state()
        if serialized_state != self.last_state_hash:
            changed = True

        if changed:
            self.last_state_hash = serialized_state
            self.last_updated_at = datetime.now().isoformat()

        return changed

    def complete_all(self) -> bool:
        """Помечает все задачи завершенными."""
        if not self.active or not self.todos:
            return False

        changed = False
        for todo in self.todos:
            if todo.status != "completed":
                todo.status = "completed"
                changed = True

        if changed:
            self._record_state_change()

        return changed

    def has_pending_tasks(self) -> bool:
        return any(todo.status != "completed" for todo in self.todos)


class SmartAgent:
    """Умный агент для решения задач пользователя с улучшенным анализом и планированием."""

    def __init__(self, gigachat_client: GigaChatClient):
        self.client = gigachat_client
        self.intent_analyzer = AdvancedIntentAnalyzer(gigachat_client)
        self.task_planner = AdvancedTaskPlanner(gigachat_client)
        
        # Инициализируем инструменты
        self.web_search = WebSearchTool()
        self.web_parser = WebParsingTool()
        self.browser = BrowserTool()
        self.code_executor = CodeExecutor()
        self.file_system = FileSystemTools()
        self.metacognition = MetacognitionManager(self.file_system)
        self.planning_tool = PlanningToolManager(self.metacognition)

        # Контроль попыток ленивого ответа без вызова инструментов
        self.lazy_response_limit = 3

        # История выполнения
        self.execution_history = []
        
    def get_available_functions(self) -> List[Dict]:
        """Возвращает список доступных функций."""
        functions = []
        
        if self.web_search.available:
            functions.append({
                "name": "web_search",
                "description": "Выполняет поиск в интернете по запросу",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Поисковый запрос"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Максимальное количество результатов",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            })
        
        if self.web_parser.available:
            functions.append({
                "name": "web_parse",
                "description": "Извлекает содержимое веб-страницы",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL страницы для парсинга"
                        },
                        "extract_focus": {
                            "type": "string",
                            "description": "Фокус для извлечения (ключевые слова)"
                        }
                    },
                    "required": ["url"]
                }
            })
        
        if self.browser.available:
            functions.extend([
                {
                    "name": "browser_navigate",
                    "description": "Переходит на веб-страницу в браузере с поддержкой динамического контента",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL для перехода"
                            }
                        },
                        "required": ["url"]
                    }
                },
                {
                    "name": "browser_extract",
                    "description": "Извлекает контент со страницы в браузере с ожиданием динамического контента",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "CSS селектор для извлечения (опционально)"
                            },
                            "wait_for_element": {
                                "type": "boolean",
                                "description": "Ждать появления элемента",
                                "default": True
                            }
                        }
                    }
                },
                {
                    "name": "browser_click",
                    "description": "Кликает по элементу на странице",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {
                                "type": "string",
                                "description": "Селектор элемента для клика"
                            }
                        },
                        "required": ["selector"]
                    }
                },
                {
                    "name": "wait_dynamic_content",
                    "description": "Ждет загрузки динамического контента на странице",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timeout": {
                                "type": "integer",
                                "description": "Таймаут ожидания в миллисекундах",
                                "default": 10000
                            }
                        }
                    }
                }
            ])

        functions.append({
            "name": "code_execute",
            "description": "Выполняет Python код для вычислений и автоматизации",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python код для выполнения"
                    }
                },
                "required": ["code"]
            }
        })

        # Реальные инструменты файловой системы для метакогнитивной памяти
        functions.extend([
            {
                "name": "ls",
                "description": "Показывает содержимое рабочей директории агента",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Путь относительно рабочей директории (опционально)"
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Показывать содержимое рекурсивно",
                            "default": False
                        },
                        "include_hidden": {
                            "type": "boolean",
                            "description": "Включать скрытые файлы",
                            "default": False
                        }
                    }
                }
            },
            {
                "name": "read_file",
                "description": "Читает файл с номерами строк из рабочей директории",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Путь к файлу относительно рабочей директории"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "С какой строки начать чтение",
                            "default": 0
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Сколько строк прочитать",
                            "default": 2000
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": "Создает или перезаписывает файл в рабочей директории",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Путь к файлу относительно рабочей директории"
                        },
                        "content": {
                            "type": "string",
                            "description": "Содержимое файла"
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "Перезаписать файл, если он существует",
                            "default": False
                        }
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "edit_file",
                "description": "Заменяет текст в существующем файле",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Путь к файлу относительно рабочей директории"
                        },
                        "old_string": {
                            "type": "string",
                            "description": "Исходный текст для замены"
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Новый текст"
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "Заменить все вхождения",
                            "default": False
                        }
                    },
                    "required": ["path", "old_string", "new_string"]
                }
            }
        ])

        functions.append({
            "name": "write_todos",
            "description": "Обновляет todo-список (Planning Tool) для сложных задач. Передавай полный актуальный список задач со статусами.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "Список задач, который нужно зафиксировать",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Описание задачи"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "Статус задачи"
                                }
                            },
                            "required": ["content", "status"]
                        }
                    }
                },
                "required": ["todos"]
            }
        })

        functions.append({
            "name": "finish_task",
            "description": "Завершает выполнение задачи с финальным ответом",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Финальный ответ пользователю"
                    }
                },
                "required": ["answer"]
            }
        })
        
        return functions
    
    def execute_function(self, function_name: str, arguments: Dict) -> ToolResult:
        """Выполняет функцию."""
        try:
            arguments = arguments or {}

            if function_name == "web_search":
                return self.web_search.search(
                    query=arguments.get("query"),
                    max_results=arguments.get("max_results", 5)
                )
            
            elif function_name == "web_parse":
                return self.web_parser.parse(
                    url=arguments.get("url"),
                    extract_focus=arguments.get("extract_focus")
                )
            
            elif function_name == "browser_navigate":
                return self.browser.navigate(url=arguments.get("url"))
            
            elif function_name == "browser_extract":
                return self.browser.extract_content(
                    selector=arguments.get("selector"),
                    wait_for_element=arguments.get("wait_for_element", True)
                )
            
            elif function_name == "browser_click":
                return self.browser.click_element(selector=arguments.get("selector"))
            
            elif function_name == "wait_dynamic_content":
                return self.browser.wait_for_dynamic_content(
                    timeout=arguments.get("timeout", 10000)
                )

            elif function_name == "code_execute":
                return self.code_executor.execute(code=arguments.get("code"))
            
            elif function_name == "ls":
                return self.file_system.list_files(
                    path=arguments.get("path"),
                    recursive=arguments.get("recursive", False),
                    include_hidden=arguments.get("include_hidden", False)
                )

            elif function_name == "read_file":
                return self.file_system.read_file(
                    file_path=arguments.get("path"),
                    offset=arguments.get("offset", 0),
                    limit=arguments.get("limit", 2000)
                )

            elif function_name == "write_file":
                return self.file_system.write_file(
                    file_path=arguments.get("path"),
                    content=arguments.get("content", ""),
                    overwrite=arguments.get("overwrite", False)
                )

            elif function_name == "edit_file":
                return self.file_system.edit_file(
                    file_path=arguments.get("path"),
                    old_string=arguments.get("old_string", ""),
                    new_string=arguments.get("new_string", ""),
                    replace_all=arguments.get("replace_all", False)
                )

            elif function_name == "write_todos":
                return self.planning_tool.handle_write_todos(arguments.get("todos"))

            elif function_name == "finish_task":
                return ToolResult(
                    tool_name="finish_task",
                    success=True,
                    data=arguments.get("answer"),
                    confidence=1.0
                )
            
            else:
                return ToolResult(
                    tool_name=function_name,
                    success=False,
                    data=None,
                    error=f"Неизвестная функция: {function_name}"
                )

        except Exception as e:
            return ToolResult(
                tool_name=function_name,
                success=False,
                data=None,
                error=str(e)
            )

    def _initialize_plan_tracking(self, plan: ExecutionPlan) -> None:
        """Подготавливает план к отслеживанию выполнения длинных цепочек."""
        plan.current_step_index = 0
        plan.completed_steps = 0
        plan.progress_notes = []
        plan.progress = {
            'total_steps': len(plan.steps),
            'completed': 0,
            'failed': 0,
            'unplanned_calls': 0,
            'history': [],
            'current_step_order': 0
        }

        for order, step in enumerate(plan.steps):
            step['order'] = order
            step['status'] = 'pending'
            step['attempts'] = 0
            step['history'] = []
            step['last_result_summary'] = None

    def _get_next_pending_step(self, plan: ExecutionPlan) -> Optional[Dict[str, Any]]:
        """Возвращает следующий незавершенный шаг плана."""
        for step in plan.steps:
            if step.get('status') == 'pending':
                return step
        return None

    def _summarize_result(self, result: ToolResult, max_length: int = 200) -> str:
        """Создает краткую сводку по результату инструмента."""
        status = "успешно" if result.success else f"ошибка: {result.error}"

        data_preview = ""
        if result.data:
            data_str = str(result.data).strip()
            if len(data_str) > max_length:
                data_str = data_str[:max_length] + "..."
            data_preview = f" | данные: {data_str}"

        metadata_preview = ""
        if result.metadata:
            items = list(result.metadata.items())[:3]
            if items:
                metadata_pairs = ", ".join(f"{key}={value}" for key, value in items)
                metadata_preview = f" | метаданные: {metadata_pairs}"

        timing_info = ""
        if result.execution_time:
            timing_info = f" | время: {result.execution_time:.2f}с"

        confidence_info = ""
        if result.confidence:
            confidence_info = f" | уверенность: {result.confidence:.0%}"

        summary = f"{status}{data_preview}{metadata_preview}{timing_info}{confidence_info}".strip()
        return summary

    def _update_plan_progress(self, plan: ExecutionPlan, tool_name: str, result: ToolResult) -> Dict[str, Any]:
        """Обновляет прогресс плана после очередного шага."""
        if not hasattr(plan, 'progress'):
            self._initialize_plan_tracking(plan)

        progress = plan.progress
        summary = self._summarize_result(result)
        planned = False
        repeated_execution = False
        step_order = None

        for step in plan.steps:
            if step.get('tool') == tool_name and step.get('status') == 'pending':
                planned = True
                step_order = step.get('order')
                step['attempts'] = step.get('attempts', 0) + 1
                step_history = step.setdefault('history', [])
                step_history.append({
                    'success': result.success,
                    'summary': summary,
                    'timestamp': datetime.now().isoformat()
                })
                if len(step_history) > 10:
                    step['history'] = step_history[-10:]

                step['last_result_summary'] = summary

                if result.success:
                    step['status'] = 'completed'
                    progress['completed'] = progress.get('completed', 0) + 1
                    plan.completed_steps = progress['completed']
                else:
                    step['status'] = 'pending'
                    step['last_error'] = result.error
                    progress['failed'] = progress.get('failed', 0) + 1
                break
        else:
            for step in plan.steps:
                if step.get('tool') == tool_name:
                    planned = True
                    repeated_execution = True
                    step_order = step.get('order')
                    step_history = step.setdefault('history', [])
                    step_history.append({
                        'success': result.success,
                        'summary': summary,
                        'timestamp': datetime.now().isoformat(),
                        'note': 'повторное выполнение'
                    })
                    if len(step_history) > 10:
                        step['history'] = step_history[-10:]
                    break

        status_label = 'success' if result.success else 'failed'
        progress_history = progress.setdefault('history', [])
        progress_history.append({
            'tool': tool_name,
            'status': status_label if planned else 'unplanned',
            'summary': summary,
            'timestamp': datetime.now().isoformat(),
            'step_order': step_order
        })
        if len(progress_history) > 30:
            progress['history'] = progress_history[-30:]

        note = None
        if planned:
            if repeated_execution:
                note = f"Инструмент {tool_name} выполнен повторно после шага {step_order + 1 if step_order is not None else '?'}"
            elif result.success:
                note = f"Шаг {step_order + 1 if step_order is not None else '?'} ({tool_name}) выполнен успешно"
            else:
                note = f"Шаг {step_order + 1 if step_order is not None else '?'} ({tool_name}) завершился ошибкой"
        else:
            progress['unplanned_calls'] = progress.get('unplanned_calls', 0) + 1
            note = f"Выполнен внеплановый инструмент {tool_name}"

        if note:
            plan.progress_notes.append(note)
            if len(plan.progress_notes) > 10:
                plan.progress_notes = plan.progress_notes[-10:]

        next_pending = self._get_next_pending_step(plan)
        if next_pending:
            plan.current_step_index = next_pending.get('order', plan.current_step_index)
            progress['current_step_order'] = plan.current_step_index
        else:
            plan.current_step_index = len(plan.steps)
            progress['current_step_order'] = plan.current_step_index

        plan.progress = progress

        return {
            'planned': planned,
            'step_order': step_order,
            'summary': summary,
            'note': note,
            'status': status_label if planned else 'unplanned'
        }

    def _build_plan_progress_payload(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Формирует краткое описание прогресса плана."""
        if not hasattr(plan, 'progress'):
            self._initialize_plan_tracking(plan)

        progress = plan.progress
        pending_steps = []
        for step in plan.steps:
            if step.get('status') == 'pending':
                pending_steps.append({
                    'order': step.get('order'),
                    'tool': step.get('tool'),
                    'description': step.get('description'),
                    'expected_outcome': step.get('expected_outcome')
                })

        next_step = pending_steps[0] if pending_steps else None

        planning_state = self.planning_tool.get_state() if hasattr(self, "planning_tool") and self.planning_tool else {
            "active": False,
            "todos": [],
            "last_updated_at": None
        }

        return {
            'total_steps': progress.get('total_steps', len(plan.steps)),
            'completed_steps': progress.get('completed', plan.completed_steps),
            'failed_attempts': progress.get('failed', 0),
            'unplanned_calls': progress.get('unplanned_calls', 0),
            'current_step_order': progress.get('current_step_order', plan.current_step_index),
            'next_step': next_step,
            'pending_steps': pending_steps[:5],
            'recent_history': progress.get('history', [])[-5:],
            'recent_notes': plan.progress_notes[-5:] if plan.progress_notes else [],
            'planning_tool': planning_state
        }

    def _has_pending_plan_steps(self, plan: ExecutionPlan) -> bool:
        """Проверяет, остались ли невыполненные шаги плана."""
        if not plan or not getattr(plan, 'steps', None):
            return False

        for step in plan.steps:
            tool_name = step.get('tool')
            if tool_name == 'finish_task':
                continue
            status = step.get('status')
            if status not in ('completed', 'skipped'):
                return True
        return False

    def _task_requires_active_tools(self, context: TaskContext) -> bool:
        """Определяет, требует ли задача обязательного использования инструментов."""
        return any([
            context.requires_search,
            context.requires_browser,
            context.requires_computation,
            context.requires_terminal
        ])

    def _should_force_tool_usage(self, context: TaskContext, plan: ExecutionPlan) -> bool:
        """Нужно ли настоятельно требовать вызова инструментов вместо текстового ответа."""
        return self._task_requires_active_tools(context) and self._has_pending_plan_steps(plan)

    def _build_tool_usage_reminder(self, context: TaskContext, plan: ExecutionPlan, attempt: int) -> str:
        """Формирует напоминание модели о необходимости вызвать инструменты."""
        pending_tools = [
            step.get('tool', 'неизвестный инструмент')
            for step in plan.steps
            if step.get('status') == 'pending'
        ] if getattr(plan, 'steps', None) else []

        if not pending_tools and getattr(plan, 'steps', None):
            pending_tools = [step.get('tool', 'неизвестный инструмент') for step in plan.steps]

        tools_preview = ", ".join(pending_tools[:5]) if pending_tools else "запланированные инструменты"

        focus_clauses: List[str] = []
        if context.focus_points:
            focus_clauses.append("ключевые шаги: " + "; ".join(context.focus_points[:2]))
        if context.verification_checks:
            focus_clauses.append("проверки: " + "; ".join(context.verification_checks[:2]))
        if context.output_expectations:
            focus_clauses.append("итог: " + "; ".join(context.output_expectations[:1]))

        focus_hint = ""
        if focus_clauses:
            focus_hint = " Помни про " + " | ".join(focus_clauses) + "."

        reminder = (
            f"⚠️ Попытка №{attempt}: Нельзя ограничиваться описанием шагов. "
            f"Выполни реальные вызовы функций согласно плану ({tools_preview}) и предоставь результаты каждого инструмента. "
            "Не завершай задачу до фактического выполнения всех необходимых инструментов и вызова finish_task только по завершении." +
            focus_hint
        )
        return reminder

    def _build_focus_guardrail(self, context: TaskContext, reason: Optional[str] = None) -> str:
        """Создает системное напоминание о фокусе на намерениях пользователя."""

        lines: List[str] = ["🔁 Фокус на намерениях пользователя."]
        if reason:
            lines[0] += f" Причина: {reason}."

        if context.primary_objective:
            lines.append(f"Цель: {context.primary_objective}.")

        if context.focus_points:
            lines.append("Ключевые действия: " + "; ".join(context.focus_points[:2]) + ".")

        if context.verification_checks:
            lines.append("Проверки: " + "; ".join(context.verification_checks[:2]) + ".")

        if context.output_expectations:
            lines.append("Итог: " + "; ".join(context.output_expectations[:1]) + ".")

        if context.prohibited_actions:
            lines.append("Запреты: " + "; ".join(context.prohibited_actions[:1]) + ".")

        lines.append("Всегда подтверждай данные ссылками на официальный источник и избегай догадок.")

        return "\n".join(lines)

    def _format_plan_steps(self, steps):
        """Форматирует шаги плана в читаемый вид."""
        formatted_steps = []
        for i, step in enumerate(steps, 1):
            description = step.get('description', step['tool'])
            priority = step.get('priority', 'не указан')
            formatted_steps.append(f"{i}. {description} (приоритет: {priority})")
        return "\n".join(formatted_steps)
    
    def build_enhanced_system_prompt(self, context: TaskContext, plan: ExecutionPlan) -> str:
        """Создает улучшенный системный промпт с метаролью и я-конструкциями."""
        
        tools_status = f"""
МОИ ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
🔍 Веб-поиск: {'✅ Работает' if self.web_search.available else '❌ Недоступен'}
📄 Парсинг страниц: {'✅ Работает' if self.web_parser.available else '❌ Недоступен'}
🌐 Браузер: {'✅ Работает' if self.browser.available else '❌ Недоступен'}
💻 Выполнение кода и команд: ✅ Работает
🗂️ Файловая память: ✅ Работает (каталог: {self.file_system.base_dir})"""

        plan_reasoning = plan.reasoning if plan.reasoning else "План создан на основе универсальных инструментов"

        file_system_guidelines = f"""
МОЯ ФАЙЛОВАЯ ПАМЯТЬ:
- Рабочая директория: {self.file_system.base_dir}
- Всегда придерживаюсь последовательности ls → read_file перед write_file или edit_file
- Храню заметки и журналы задач в каталоге metacognition/ для устойчивой метапамяти"""

        # Исправляем строку с форматированием
        plan_steps_formatted = '\n'.join([f'{i}. {step.get("description", step["tool"])} (приоритет: {step.get("priority", "не указан")})' for i, step in enumerate(plan.steps, 1)])

        focus_lines: List[str] = []

        if context.primary_objective:
            focus_lines.append(f"🎯 Главная цель: {context.primary_objective}")

        if context.focus_summary and context.focus_summary not in focus_lines:
            focus_lines.append(f"🧭 Обзор: {context.focus_summary}")

        if context.focus_points:
            focus_points_preview = context.focus_points[:5]
            focus_lines.append(
                "🔑 Ключевые действия:\n  - " + "\n  - ".join(focus_points_preview)
            )

        if context.verification_checks:
            verification_preview = context.verification_checks[:3]
            focus_lines.append(
                "✅ Обязательные проверки:\n  - " + "\n  - ".join(verification_preview)
            )

        if context.output_expectations:
            output_preview = context.output_expectations[:3]
            focus_lines.append(
                "📦 Итоговые артефакты:\n  - " + "\n  - ".join(output_preview)
            )

        if context.prohibited_actions:
            prohibited_preview = context.prohibited_actions[:3]
            focus_lines.append(
                "⛔ Запреты:\n  - " + "\n  - ".join(prohibited_preview)
            )

        focus_section = ""
        if focus_lines:
            focus_section = "МОЙ ФОКУС НА НАМЕРЕНИЯХ:\n" + "\n".join(focus_lines)

        return f"""Я - продвинутый интеллектуальный агент X-Master v77 Enhanced. Моя роль - эффективно решать задачи пользователей, используя доступные инструменты и глубокий анализ.

МОЯ ТЕКУЩАЯ СИТУАЦИЯ:
📅 Сегодня: {CURRENT_DATE_FORMATTED}
🎯 Задача пользователя: "{context.query}"
🧠 Мой анализ намерений: {context.intent} (уверенность: {context.confidence_score:.1%})
📋 Цель пользователя: {context.user_goal}
⚡ Сложность: {context.complexity}
🌍 Область: {context.domain}
⏰ Срочность: {context.urgency}
🕐 Временной контекст: {context.temporal_context}

{tools_status}

{focus_section if focus_section else ''}

{file_system_guidelines}

МОЙ ПЛАН ДЕЙСТВИЙ:
{plan_reasoning}

Конкретные шаги:
{plan_steps_formatted}

  Критерии успеха: {', '.join(plan.success_criteria) if plan.success_criteria else 'Полный и точный ответ пользователю'}

  МОЙ ПЛАНИРОВЩИК ЗАДАЧ (PLANNING TOOL):
  - Для сложных или многошаговых задач и по запросу пользователя я веду todo-список через write_todos
  - Первую актуальную задачу отмечаю in_progress, остальные pending и обновляю статусы сразу после прогресса
  - Поддерживаю только статусы pending/in_progress/completed и не допускаю отсутствия активной задачи
  - При изменении плана адаптирую список: удаляю лишние пункты, добавляю новые и фиксирую фактический прогресс
  - Todo-список показывает пользователю мой ход работы, поэтому обновляю его оперативно

  МОИ ПРИНЦИПЫ РАБОТЫ:
1. Я всегда начинаю с глубокого анализа поставленной задачи
2. Я использую инструменты последовательно и обдуманно
3. Я проверяю результаты каждого шага перед переходом к следующему
4. Для динамических сайтов я обязательно жду загрузки контента
5. Я синтезирую информацию из разных источников для полного ответа
6. Я учитываю текущую дату при поиске актуальной информации
7. Я ОБЯЗАТЕЛЬНО завершаю каждую задачу вызовом finish_task с исчерпывающим ответом
8. Я отслеживаю прогресс длинных цепочек инструментов и последовательно выполняю шаги плана
9. Мне запрещено заменять реальные вызовы инструментов описанием действий — если план не выполнен, я продолжаю использовать инструменты до завершения

МОИ МЕТАКОГНИТИВНЫЕ СПОСОБНОСТИ:
- Я анализирую свои действия и корректирую план при необходимости
- Я понимаю ограничения своих инструментов и адаптируюсь
- Я оцениваю качество получаемых данных и ищу дополнительные источники
- Я помню контекст всего диалога и использую накопленную информацию
- Я использую файловую систему для фиксации анализа, планов и результатов

МОЙ МОНИТОРИНГ ПРОГРЕССА:
- Я фиксирую статус каждого шага плана и отмечаю завершенные инструменты
- После каждого шага я напоминаю себе, какой инструмент идет следующим
- Для длинных последовательностей я регулярно сверяюсь с планом и избегаю пропусков

СТРАТЕГИЯ РАБОТЫ С ВРЕМЕННЫМИ ДАННЫМИ:
Поскольку сегодня {CURRENT_DATE_FORMATTED}, я:
- Добавляю текущую дату к поисковым запросам об актуальных событиях
- Использую браузер для получения свежих данных с официальных сайтов
- Указываю дату получения информации в своих ответах
- Проверяю актуальность найденной информации

Теперь я приступаю к выполнению задачи с полной концентрацией и профессионализмом!"""
    
    def process_query(self, query: str, max_iterations: int = 15) -> Dict[str, Any]:
        """Обрабатывает запрос пользователя с улучшенным анализом."""
        logger.info(f"Начинаю обработку запроса: {query}")
        logger.info(f"Текущая дата для контекста: {CURRENT_DATE_FORMATTED}")

        # Сбрасываем состояние Planning Tool для новой задачи
        if hasattr(self, "planning_tool") and self.planning_tool:
            self.planning_tool.reset()

        # Глубокий анализ намерений с помощью LLM
        context = self.intent_analyzer.analyze_with_llm(query)
        logger.info(f"Результат анализа намерений: {context}")

        # Создаем умный план выполнения
        plan = self.task_planner.create_smart_plan(context)
        logger.info(f"Создан план: {[step['tool'] for step in plan.steps]}")
        logger.info(f"Обоснование плана: {plan.reasoning}")

        # Подготавливаем план к отслеживанию длительных цепочек
        self._initialize_plan_tracking(plan)

        planning_tool_auto_started = False
        if hasattr(self, "planning_tool") and self.planning_tool.should_activate(context, plan):
            initial_todos = self.planning_tool.initialize_from_plan(plan)
            if initial_todos:
                planning_tool_auto_started = True
                logger.info(
                    "Planning Tool активирован автоматически: сформировано %s задач",
                    len(initial_todos)
                )
                plan.progress_notes.append("Активирован Planning Tool: сформирован todo-список по плану")
                if len(plan.progress_notes) > 10:
                    plan.progress_notes = plan.progress_notes[-10:]
            else:
                logger.debug("Planning Tool не активирован: автоматическая инициализация не создала задач")
        else:
            logger.debug("Planning Tool не требует активации для данной задачи")

        # Запускаем файловую метапамять
        session_dir = self.metacognition.start_session(query, context, plan)
        plan_progress_payload = self._build_plan_progress_payload(plan)
        if session_dir:
            self.metacognition.update_progress(plan_progress_payload)

        # Инициализируем диалог с улучшенным промптом
        system_prompt = self.build_enhanced_system_prompt(context, plan)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        self.metacognition.record_system_prompt(system_prompt)

        execution_log = []
        final_answer = None
        lazy_response_attempts = 0
        focus_reinforcements = 0
        max_focus_reinforcements = 3
        functions_state_id: Optional[str] = None

        iteration_limit = max(max_iterations, len(plan.steps) * 2 if plan.steps else max_iterations)
        if iteration_limit != max_iterations:
            logger.info(f"Увеличен лимит итераций до {iteration_limit} для длинной цепочки инструментов")

        for iteration in range(iteration_limit):
            try:
                logger.info(f"Итерация {iteration + 1}/{iteration_limit}")
                
                # Отправляем запрос в GigaChat
                response = self.client.chat(
                    messages=messages,
                    functions=self.get_available_functions(),
                    temperature=0.2,  # Снижаем температуру для более точного следования плану
                    function_call="auto",
                    functions_state_id=functions_state_id
                )
                
                if not response or 'choices' not in response:
                    logger.error("Некорректный ответ от GigaChat")
                    break
                
                choice = response['choices'][0]
                message = choice['message']
                functions_state_id = (
                    message.get('functions_state_id')
                    or choice.get('functions_state_id')
                    or response.get('functions_state_id')
                    or functions_state_id
                )

                # Добавляем сообщение в диалог
                assistant_message: Dict[str, Any] = {"role": message['role']}
                content = message.get('content')
                assistant_message["content"] = content if content is not None else ""
                if message.get('function_call'):
                    assistant_message["function_call"] = message.get('function_call')
                messages.append(assistant_message)
                
                # Обрабатываем вызов функции
                if 'function_call' in message:
                    func_call = message['function_call']
                    func_name = func_call.get('name')
                    
                    # Парсим аргументы
                    try:
                        if isinstance(func_call.get('arguments'), str):
                            func_args = json.loads(func_call['arguments'])
                        else:
                            func_args = func_call.get('arguments', {})
                    except json.JSONDecodeError:
                        func_args = {}
                    
                    logger.info(f"Выполняется функция: {func_name} с аргументами: {func_args}")

                    # Выполняем функцию и обновляем прогресс плана
                    result = self.execute_function(func_name, func_args)
                    lazy_response_attempts = 0
                    progress_info = self._update_plan_progress(plan, func_name, result)
                    planning_tool_changed = self.planning_tool.sync_with_plan(plan)
                    if planning_tool_changed:
                        plan.progress_notes.append(f"Planning Tool обновлен после вызова {func_name}")
                        if len(plan.progress_notes) > 10:
                            plan.progress_notes = plan.progress_notes[-10:]

                    guardrail_reason: Optional[str] = None
                    guardrail_message: Optional[str] = None
                    if not progress_info.get('planned', False):
                        guardrail_reason = 'внеплановый вызов инструмента'
                    elif not result.success:
                        guardrail_reason = 'ошибка инструмента'

                    if guardrail_reason and focus_reinforcements < max_focus_reinforcements:
                        guardrail_message = self._build_focus_guardrail(context, guardrail_reason)
                        focus_reinforcements += 1
                        logger.warning(
                            "Активировано напоминание о фокусе (причина: %s, итого: %s)",
                            guardrail_reason,
                            focus_reinforcements
                        )
                        plan.progress_notes.append(f"Фокус-намерение усилено: {guardrail_reason}")
                        if len(plan.progress_notes) > 10:
                            plan.progress_notes = plan.progress_notes[-10:]

                    plan_progress_payload = self._build_plan_progress_payload(plan)

                    execution_log.append({
                        'function': func_name,
                        'arguments': func_args,
                        'result': result,
                        'iteration': iteration + 1,
                        'timestamp': datetime.now().isoformat(),
                        'planned': progress_info.get('planned', False),
                        'plan_step_order': progress_info.get('step_order'),
                        'plan_status': progress_info.get('status'),
                        'plan_note': progress_info.get('note'),
                        'plan_summary': progress_info.get('summary'),
                        'plan_state': {
                            'completed_steps': plan_progress_payload.get('completed_steps'),
                            'pending_steps': len(plan_progress_payload.get('pending_steps', [])),
                            'next_tool': plan_progress_payload.get('next_step', {}).get('tool') if plan_progress_payload.get('next_step') else None
                        }
                    })

                    self.metacognition.log_tool_execution(
                        tool_name=func_name,
                        arguments=func_args,
                        result=result,
                        plan_progress=plan_progress_payload,
                        note=progress_info.get('note')
                    )

                    # Добавляем результат в диалог с дополнительным контекстом
                    function_payload = {
                        "success": result.success,
                        "data": str(result.data)[:2000] if result.data is not None else None,
                        "error": result.error,
                        "metadata": result.metadata,
                        "confidence": result.confidence,
                        "execution_time": result.execution_time,
                        "plan_progress": plan_progress_payload,
                        "plan_note": progress_info.get('note'),
                        "plan_summary": progress_info.get('summary')
                    }
                    try:
                        function_content = json.dumps(function_payload, ensure_ascii=False, default=str)
                    except TypeError:
                        sanitized_payload = {key: str(value) for key, value in function_payload.items()}
                        function_content = json.dumps(sanitized_payload, ensure_ascii=False)

                    function_response = {
                        "role": "function",
                        "name": func_name or "unknown_function",
                        "content": function_content
                    }
                    messages.append(function_response)

                    if guardrail_message:
                        if messages and messages[0].get("role") == "system":
                            existing_content = messages[0].get("content", "")
                            separator = "\n\n" if existing_content else ""
                            messages[0]["content"] = f"{existing_content}{separator}{guardrail_message}"
                        else:
                            messages.insert(0, {"role": "system", "content": guardrail_message})

                    # Проверяем на завершение задачи
                    if func_name == "finish_task" and result.success:
                        final_answer = result.data
                        break
                
                else:
                    # Модель ответила без вызова функции
                    content = (message.get('content') or '').strip()

                    if not content:
                        logger.debug("Получен пустой ответ без вызова функции, ожидаю дальнейших действий модели")
                        continue

                    if self._should_force_tool_usage(context, plan) and not final_answer:
                        lazy_response_attempts += 1
                        reminder_message = self._build_tool_usage_reminder(context, plan, lazy_response_attempts)
                        logger.warning(
                            "Модель попыталась завершить задачу без инструментов (попытка %s). Отправлено напоминание.",
                            lazy_response_attempts
                        )

                        reminder_block = (
                            f"\n\n[СИСТЕМНОЕ НАПОМИНАНИЕ #{lazy_response_attempts}] "
                            f"{reminder_message}"
                        )

                        if messages and messages[0].get("role") == "system":
                            messages[0]["content"] = (
                                messages[0].get("content", "") + reminder_block
                            )
                        else:
                            messages.insert(0, {
                                "role": "system",
                                "content": reminder_message
                            })

                        note_text = (
                            f"Напоминание о необходимости вызвать инструменты (попытка {lazy_response_attempts})"
                        )
                        plan.progress_notes.append(note_text)
                        if len(plan.progress_notes) > 10:
                            plan.progress_notes = plan.progress_notes[-10:]

                        if lazy_response_attempts >= self.lazy_response_limit:
                            logger.warning(
                                "Достигнут предел напоминаний о вызове инструментов (%s попыток)",
                                self.lazy_response_limit
                            )
                        continue

                    if content and not final_answer:
                        # Принудительно вызываем finish_task
                        result = self.execute_function("finish_task", {"answer": content})
                        lazy_response_attempts = 0
                        progress_info = self._update_plan_progress(plan, "finish_task", result)
                        planning_tool_changed = self.planning_tool.sync_with_plan(plan)
                        if planning_tool_changed:
                            plan.progress_notes.append("Planning Tool обновлен после вызова finish_task")
                            if len(plan.progress_notes) > 10:
                                plan.progress_notes = plan.progress_notes[-10:]
                        plan_progress_payload = self._build_plan_progress_payload(plan)
                        execution_log.append({
                            'function': 'finish_task',
                            'arguments': {"answer": content},
                            'result': result,
                            'iteration': iteration + 1,
                            'timestamp': datetime.now().isoformat(),
                            'planned': progress_info.get('planned', True),
                            'plan_status': progress_info.get('status', 'completion'),
                            'plan_note': progress_info.get('note') or 'Завершение задачи финальным ответом',
                            'plan_summary': progress_info.get('summary') or 'finish_task',
                            'plan_state': {
                                'completed_steps': plan_progress_payload.get('completed_steps'),
                                'pending_steps': len(plan_progress_payload.get('pending_steps', [])),
                                'next_tool': None
                            }
                        })
                        self.metacognition.log_tool_execution(
                            tool_name='finish_task',
                            arguments={"answer": content},
                            result=result,
                            plan_progress=plan_progress_payload,
                            note=progress_info.get('note') or 'Завершение задачи финальным ответом'
                        )
                        plan.progress_notes.append('Задача закрыта вызовом finish_task')
                        if len(plan.progress_notes) > 10:
                            plan.progress_notes = plan.progress_notes[-10:]
                        final_answer = content
                        break
                
            except Exception as e:
                logger.error(f"Ошибка в итерации {iteration + 1}: {e}")
                break
        
        # Закрываем браузер если был открыт
        try:
            if self.browser.page:
                self.browser.close_session()
        except:
            pass

        # Финально синхронизируем Planning Tool
        self.planning_tool.sync_with_plan(plan)
        if final_answer and not self._has_pending_plan_steps(plan):
            self.planning_tool.complete_all()

        # Анализируем выполнение
        successful_tools = sum(1 for log in execution_log if log['result'].success)
        total_tools = len(execution_log)
        
        # Получаем списки инструментов
        planned_tools = [step['tool'] for step in plan.steps]
        executed_tools = [log['function'] for log in execution_log]

        # Анализируем следование плану
        plan_adherence = sum(1 for log in execution_log if log.get('planned', False))
        plan_adherence_percent = (plan_adherence / max(total_tools, 1)) * 100
        plan_completion_percent = (plan.completed_steps / len(plan.steps) * 100) if plan.steps else 100.0
        plan_progress_payload = self._build_plan_progress_payload(plan)

        # Сохраняем историю выполнения
        self.execution_history = execution_log

        # Количество итераций, которые фактически были использованы
        iterations_used = (iteration + 1) if 'iteration' in locals() else 0

        # Оцениваем качество выполнения
        quality_score = 0.0
        if final_answer:
            quality_score += 0.4  # Есть финальный ответ
        if successful_tools > 0:
            quality_score += 0.3 * (successful_tools / max(total_tools, 1))  # Успешность инструментов
        if plan.steps:
            completion_ratio = min(1.0, plan.completed_steps / max(len(plan.steps), 1))
            quality_score += 0.3 * completion_ratio
        elif plan_adherence_percent > 50:
            quality_score += 0.3  # Следование плану

        final_summary_payload = {
            'success': final_answer is not None,
            'final_answer_present': final_answer is not None,
            'quality_score': quality_score,
            'tools_used': total_tools,
            'successful_tools': successful_tools,
            'plan_completion_percent': plan_completion_percent,
            'plan_adherence_percent': plan_adherence_percent,
            'iterations_used': iterations_used,
            'plan_progress': plan_progress_payload,
            'planning_tool_state': self.planning_tool.get_state(),
            'planning_tool_auto_started': planning_tool_auto_started,
            'planning_tool_used': self.planning_tool.active,
            'focus_reinforcements': focus_reinforcements
        }
        self.metacognition.finalize(final_answer, final_summary_payload)

        return {
            'success': final_answer is not None,
            'final_answer': final_answer or "Не удалось получить ответ",
            'context': context,
            'plan': plan,
            'execution_log': execution_log,
            'total_iterations': iterations_used,
            'tools_used': total_tools,
            'successful_tools': successful_tools,
            'confidence': quality_score,
            'query_date': CURRENT_DATE_STR,
            'planned_tools': planned_tools,
            'executed_tools': executed_tools,
            'plan_adherence_percent': plan_adherence_percent,
            'plan_completion_percent': plan_completion_percent,
            'plan_progress': plan_progress_payload,
            'plan_followed': plan_completion_percent >= 80 and plan_adherence_percent >= 60,
            'llm_analysis_used': context.meta_analysis.get('llm_analysis', False),
            'analysis_reasoning': context.meta_analysis.get('reasoning', ''),
            'plan_reasoning': plan.reasoning,
            'quality_score': quality_score,
            'risk_assessment': plan.risk_assessment,
            'iteration_limit': iteration_limit,
            'planning_tool_state': self.planning_tool.get_state(),
            'planning_tool_auto_started': planning_tool_auto_started,
            'planning_tool_used': self.planning_tool.active,
            'focus_reinforcements_used': focus_reinforcements
        }


def main():
    """Основное Streamlit приложение."""
    st.set_page_config(
        page_title="Умный X-Master Agent v77 Enhanced+",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Умный X-Master Agent v77 Enhanced+")
    st.markdown(f"**Интеллектуальный агент с продвинутым анализом намерений и умным планированием задач**")
    st.caption(f"📅 Текущая дата: {CURRENT_DATE_FORMATTED}")

    # Боковая панель конфигурации
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Отображаем текущую дату в сайдбаре
        st.info(f"📅 **Сегодня:** {CURRENT_DATE_FORMATTED}")

        client_id = st.text_input("GigaChat Client ID", type="password")
        client_secret = st.text_input("GigaChat Client Secret", type="password")

        st.markdown("---")

        model_name = st.selectbox(
            "Модель GigaChat",
            ["GigaChat-2-Max", "GigaChat", "GigaChat-2-Pro"],
            index=0
        )
        
        max_iterations = st.slider(
            "Максимум итераций",
            min_value=5,
            max_value=20,
            value=15,
            help="Максимальное количество шагов для решения задачи"
        )
        
        st.markdown("---")
        st.subheader("🛠️ Доступные инструменты")
        
        # Показываем статус инструментов
        tools_status = [
            ("🔍 Веб-поиск", DDGS_AVAILABLE),
            ("📄 Парсинг веб-страниц", trafilatura is not None),
            ("🌐 Браузер с динамическим контентом", PLAYWRIGHT_AVAILABLE),
            ("💻 Выполнение кода", True),
            ("🧮 Научные вычисления", SKLEARN_AVAILABLE),
            ("🧠 LLM анализ намерений", True),
            ("📋 Умное планирование", True)
        ]
        
        for tool_name, available in tools_status:
            if available:
                st.success(f"✅ {tool_name}")
            else:
                st.error(f"❌ {tool_name}")
        
        st.markdown("---")
        st.info(
            "**🚀 Новые возможности v77 Enhanced+:**\n"
            "• Глубокий LLM-анализ намерений пользователя\n"
            "• Умное планирование с метарассуждениями\n"
            "• Я-конструкции в системных промптах\n"
            "• Улучшенная оценка рисков и адаптивность\n"
            "• Контекстуальное понимание временных запросов\n"
            "• Продвинутая работа с финансовыми данными\n"
            "• Автоматическое следование критериям успеха\n"
            "• Метакогнитивный самоанализ выполнения\n"
            "• Повышенная точность и надежность результатов"
        )

    # Основной интерфейс
    st.markdown("### 💬 Интерфейс запросов")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Введите ваш запрос:",
            height=120,
            placeholder="Примеры запросов:\n• Найди последние новости об искусственном интеллекте\n• Как изменилась экономика России за последний месяц?\n• Что произошло на этой неделе в мире технологий?\n• Проанализируй текущую ситуацию на рынке криптовалют\n• Какие события запланированы на сегодня в России?\n• Сравни курсы валют и подготовь выводы",
            help="Агент автоматически проанализирует ваши намерения и создаст оптимальный план решения"
        )
    
    with col2:
        st.markdown("**🎯 Примеры задач:**")
        example_queries = [
            "Последние новости ИИ",
            "Ключевая ставка ЦБ на текущую дату",
            "События недели в России",
            "Анализ криптовалют",
            "Курсы валют сегодня",
            "Технологические тренды"
        ]
        
        for i, example in enumerate(example_queries):
            if st.button(f"🔍 {example}", key=f"example_{i}", use_container_width=True):
                query = example
                st.rerun()

    # Кнопка выполнения
    execute_button = st.button(
        "🚀 Выполнить задачу", 
        type="primary", 
        use_container_width=True,
        disabled=not (client_id and client_secret and query)
    )

    if execute_button:
        try:
            # Инициализируем агента
            client = GigaChatClient(client_id, client_secret, model=model_name)
            agent = SmartAgent(client)
            
            # Прогресс выполнения
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🤖 Агент выполняет глубокий анализ и умное планирование..."):
                status_text.text("🧠 Анализ намерений с помощью LLM...")
                progress_bar.progress(10)
                
                status_text.text("📋 Создание умного плана выполнения...")
                progress_bar.progress(20)
                
                # Обрабатываем запрос
                result = agent.process_query(query, max_iterations)
                progress_bar.progress(100)
                status_text.text("✅ Задача выполнена с использованием ИИ-анализа!")

            st.markdown("---")
            
            # Результаты
            st.subheader("📋 Результат")
            
            if result['success'] and result['final_answer']:
                st.success("**Ответ агента:**")
                st.write(result['final_answer'])
            else:
                st.error("Агент не смог найти ответ на задачу")

            # Расширенные метрики выполнения
            st.markdown("---")
            st.subheader("📊 Расширенные метрики выполнения")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "🔧 Инструментов",
                    result['tools_used'],
                    f"Успешно: {result['successful_tools']}"
                )
            
            with col2:
                st.metric(
                    "🔄 Итераций",
                    result['total_iterations'],
                    f"Лимит: {result.get('iteration_limit', max_iterations)}"
                )
            
            with col3:
                st.metric(
                    "🎯 Качество",
                    f"{result['quality_score']:.1%}",
                    "Общая оценка"
                )
            
            with col4:
                st.metric(
                    "📋 Следование плану",
                    f"{result.get('plan_completion_percent', 0):.0f}%",
                    f"Согласование: {result.get('plan_adherence_percent', 0):.0f}%"
                )
            
            with col5:
                llm_status = "🧠 LLM" if result.get('llm_analysis_used', False) else "📝 Базовый"
                st.metric(
                    "🔍 Анализ намерений",
                    llm_status,
                    f"Уверенность: {result['context'].confidence_score:.1%}"
                )
            
            # Анализ намерений и планирование
            st.markdown("---")
            st.subheader("🧠 Интеллектуальный анализ и планирование")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Анализ намерений пользователя:**")
                context = result['context']
                analysis_info = {
                    "Основное намерение": context.intent,
                    "Цель пользователя": context.user_goal,
                    "Сложность задачи": context.complexity,
                    "Предметная область": context.domain,
                    "Срочность": context.urgency,
                    "Временной контекст": context.temporal_context,
                    "Ожидаемые источники": context.expected_sources,
                    "Уверенность анализа": f"{context.confidence_score:.1%}",
                    "Использован LLM": "Да" if result.get('llm_analysis_used') else "Нет"
                }
                
                for key, value in analysis_info.items():
                    st.text(f"{key}: {value}")

                if result.get('analysis_reasoning'):
                    st.markdown("**💭 Обоснование анализа:**")
                    st.text_area("", result['analysis_reasoning'], height=100, disabled=True)

                if (
                    context.focus_summary
                    or context.focus_points
                    or context.output_expectations
                    or context.verification_checks
                ):
                    st.markdown("**🔎 Фокус намерений и критерии:**")
                    if context.focus_summary:
                        st.info(context.focus_summary)
                    if context.focus_points:
                        st.markdown("- " + "\n- ".join(context.focus_points[:5]))
                    if context.verification_checks:
                        st.caption("Проверки: " + "; ".join(context.verification_checks[:3]))
                    if context.output_expectations:
                        st.caption("Итоговые артефакты: " + "; ".join(context.output_expectations[:3]))

            with col2:
                st.markdown("**📋 Умное планирование задачи:**")
                plan = result['plan']
                plan_info = {
                    "Количество шагов": len(plan.steps),
                    "Оценочное время": f"{plan.estimated_time:.1f} сек",
                    "Уверенность в плане": f"{plan.confidence:.1%}",
                    "Уровень адаптивности": plan.adaptability_level
                }

                for key, value in plan_info.items():
                    st.text(f"{key}: {value}")

                if plan.reasoning:
                    st.markdown("**🎯 Логика планирования:**")
                    st.text_area("", plan.reasoning, height=100, disabled=True, key="plan_reasoning")

            # Planning Tool overview
            st.markdown("---")
            st.subheader("🗒️ Planning Tool — список задач")
            planning_state = result.get('planning_tool_state') or {}

            if planning_state.get('active') and planning_state.get('todos'):
                status_mapping = {
                    "pending": ("⏳", "Ожидает начала"),
                    "in_progress": ("🔄", "В работе"),
                    "completed": ("✅", "Завершено")
                }

                for idx, todo in enumerate(planning_state.get('todos', []), 1):
                    status = todo.get('status', 'pending')
                    emoji, label = status_mapping.get(status, ("•", status))
                    content = todo.get('content', f"Задача {idx}")
                    st.markdown(f"{emoji} **{idx}. {content}** — {label}")

                meta_notes = []
                if planning_state.get('last_updated_at'):
                    meta_notes.append(f"Обновлено: {planning_state['last_updated_at']}")
                if result.get('planning_tool_auto_started') is not None:
                    auto_flag = "Да" if result.get('planning_tool_auto_started') else "Нет"
                    meta_notes.append(f"Автоактивация: {auto_flag}")
                if not result.get('planning_tool_used', False):
                    meta_notes.append("Статус: не используется")
                elif not meta_notes:
                    meta_notes.append("Статус: активен")

                if meta_notes:
                    st.caption(" | ".join(meta_notes))
            else:
                st.info("Planning Tool не активировался для этой задачи.")

            # Сравнение плана и выполнения
            if result.get('planned_tools') and result.get('executed_tools'):
                st.markdown("---")
                st.subheader("📋 Выполнение плана vs. Реальность")

                plan_progress = result.get('plan_progress', {})
                if plan_progress:
                    progress_summary = f"{plan_progress.get('completed_steps', 0)}/{plan_progress.get('total_steps', 0)} шагов завершено"
                    next_tool = plan_progress.get('next_step', {}).get('tool') if plan_progress.get('next_step') else None
                    if next_tool:
                        progress_summary += f", следующий инструмент: {next_tool}"
                    st.info(f"📈 Прогресс плана: {progress_summary}")

                    if plan_progress.get('recent_notes'):
                        with st.expander("🧠 Последние заметки плана", expanded=False):
                            for note in plan_progress['recent_notes']:
                                st.write(f"- {note}")

                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📝 Запланировано:**")
                    for i, tool in enumerate(result['planned_tools'], 1):
                        if tool in result['executed_tools']:
                            st.success(f"{i}. ✅ {tool}")
                        else:
                            st.error(f"{i}. ❌ {tool}")
                
                with col2:
                    st.markdown("**⚡ Выполнено:**")
                    for i, tool in enumerate(result['executed_tools'], 1):
                        log_entry = next((log for log in result['execution_log'] if log['function'] == tool), {})
                        planned_marker = "📋" if log_entry.get('planned', False) else "🔄"
                        st.success(f"{i}. {planned_marker} {tool}")
                
                with col3:
                    st.markdown("**🎯 Критерии успеха:**")
                    success_criteria = plan.success_criteria
                    if success_criteria:
                        for i, criterion in enumerate(success_criteria, 1):
                            st.info(f"{i}. {criterion}")
                    else:
                        st.info("Стандартные критерии")

            # Детали контекста
            with st.expander("🧠 Детальный анализ контекста", expanded=False):
                context = result['context']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Определенные потребности:**")
                    needs = {
                        "Поиск информации": context.requires_search,
                        "Работа с браузером": context.requires_browser,
                        "Вычисления": context.requires_computation
                    }
                    st.json(needs)
                
                with col2:
                    st.markdown("**Извлеченные ключевые слова:**")
                    if context.keywords:
                        st.write(", ".join(context.keywords))
                    else:
                        st.write("Ключевые слова не выделены")
                
                if context.meta_analysis:
                    st.markdown("**📊 Метаанализ:**")
                    st.json(context.meta_analysis)

            # План выполнения с рисками
            with st.expander("📋 Детальный план с оценкой рисков", expanded=False):
                plan = result['plan']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Запланированные шаги:**")
                    for i, step in enumerate(plan.steps, 1):
                        st.markdown(f"{i}. **{step['tool']}**")
                        if 'description' in step:
                            st.markdown(f"   📝 {step['description']}")
                        if 'priority' in step:
                            st.markdown(f"   ⚡ Приоритет: {step['priority']}")
                        if 'expected_outcome' in step:
                            st.markdown(f"   🎯 Ожидаемый результат: {step['expected_outcome']}")
                
                with col2:
                    st.markdown("**🚨 Оценка рисков:**")
                    if plan.risk_assessment:
                        st.json(plan.risk_assessment)
                    else:
                        st.info("Анализ рисков не проводился")
                
                if plan.fallback_plan:
                    st.markdown("**🔄 Резервный план:**")
                    for i, step in enumerate(plan.fallback_plan, 1):
                        st.markdown(f"{i}. {step.get('tool', 'Неизвестный инструмент')}")

            # Детальный лог выполнения
            with st.expander("📜 Детальный лог выполнения", expanded=False):
                if result['execution_log']:
                    plan_total_steps = result.get('plan_progress', {}).get('total_steps', len(result.get('planned_tools', [])))
                    for i, log_entry in enumerate(result['execution_log'], 1):
                        func_result = log_entry['result']
                        
                        # Заголовок шага
                        success_emoji = "✅" if func_result.success else "❌"
                        planned_emoji = "📋" if log_entry.get('planned', False) else "🔄"
                        st.markdown(f"### {success_emoji} {planned_emoji} Шаг {i}: {log_entry['function']}")
                        st.caption(f"⏰ Время: {log_entry.get('timestamp', 'Не указано')}")
                        
                        # Информация о выполнении
                        exec_col1, exec_col2 = st.columns(2)
                        
                        with exec_col1:
                            # Аргументы
                            if log_entry['arguments']:
                                st.markdown("**📝 Аргументы:**")
                                st.json(log_entry['arguments'])
                        
                        with exec_col2:
                            # Результат
                            if func_result.success:
                                st.success(f"✅ Успешно (уверенность: {func_result.confidence:.1%})")
                                if func_result.execution_time:
                                    st.caption(f"⏱️ Время выполнения: {func_result.execution_time:.2f}с")
                            else:
                                st.error(f"❌ Ошибка: {func_result.error}")
                        
                        # Данные результата
                        if func_result.data:
                            data_preview = str(func_result.data)[:300]
                            if len(str(func_result.data)) > 300:
                                data_preview += "..."
                            st.text_area(f"📊 Результат:", data_preview, height=100, disabled=True, key=f"result_{i}")
                        
                        # Метаданные
                        if func_result.metadata:
                            st.markdown("**🔍 Метаданные:**")
                            st.json(func_result.metadata)

                        if log_entry.get('plan_note'):
                            st.caption(f"📈 План: {log_entry['plan_note']}")

                        if log_entry.get('plan_state'):
                            completed = log_entry['plan_state'].get('completed_steps', 0)
                            pending = log_entry['plan_state'].get('pending_steps', 0)
                            next_tool = log_entry['plan_state'].get('next_tool')
                            plan_state_text = f"{completed}/{plan_total_steps} шагов выполнено, осталось {pending}"
                            if next_tool:
                                plan_state_text += f", следующий инструмент: {next_tool}"
                            st.caption(f"🧭 Прогресс плана: {plan_state_text}")

                        st.markdown("---")
                else:
                    st.info("Функции не вызывались")

        except Exception as e:
            st.error(f"⚠️ Ошибка выполнения: {str(e)}")
            logger.error("Ошибка выполнения задачи", exc_info=True)


if __name__ == "__main__":
    main()