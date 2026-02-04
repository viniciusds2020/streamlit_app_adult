"""
Módulo de extração de informações de receituários médicos.
Implementa estratégia em camadas para economia de tokens:
1. OCR local (docling/pytesseract/easyocr)
2. Extração com regex
3. Fallback com Gemini (mais econômico)
4. Fallback final com Claude (mais preciso)
"""

import base64
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ExtractionResult:
    """Resultado da extração com metadados."""
    dados_estruturados: Dict[str, Any]
    resumo: str
    metodo_usado: str
    ocr_backend: Optional[str] = None
    texto_ocr: Optional[str] = None
    confianca: float = 0.0
    tokens_estimados: int = 0
    custo_estimado: float = 0.0


def encode_image_to_base64(image_path: str) -> str:
    """Codifica uma imagem em base64."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Retorna o media type baseado na extensão do arquivo."""
    extension = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(extension, "image/jpeg")


def _merge_data(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Mescla dados, preenchendo campos vazios/null."""
    result = base.copy()

    for key, value in updates.items():
        if value is None:
            continue

        if key not in result or result[key] is None:
            result[key] = value
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_data(result[key], value)
        elif isinstance(value, list) and isinstance(result.get(key), list):
            # Para listas, adicionar itens que não existem
            existing = {json.dumps(item, sort_keys=True) for item in result[key] if item}
            for item in value:
                if item and json.dumps(item, sort_keys=True) not in existing:
                    result[key].append(item)

    return result


def _build_complete_structure(partial_data: Dict[str, Any]) -> Dict[str, Any]:
    """Constrói estrutura completa com campos padrão."""
    template = {
        "medico": {
            "nome": None,
            "crm": None,
            "especialidade": None,
            "email": None,
            "telefones": [],
            "endereco": None,
            "website": None
        },
        "paciente": {
            "nome": None
        },
        "documento": {
            "tipo": None,
            "data": None
        },
        "prescricoes": [],
        "indicacao_clinica": None,
        "observacoes": None,
        "historico_relevante": None
    }

    return _merge_data(template, partial_data)


def process_prescription_optimized(
    image_data: bytes,
    media_type: str,
    anthropic_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    force_method: Optional[str] = None
) -> Dict[str, Any]:
    """
    Processa receituário usando estratégia em camadas para economia.

    Camadas (em ordem):
    1. OCR local + Regex (custo zero de API)
    2. Gemini com texto OCR (baixo custo)
    3. Gemini Vision (médio custo)
    4. Claude Vision (maior custo, maior precisão)

    Args:
        image_data: Bytes da imagem
        media_type: Tipo MIME da imagem
        anthropic_key: Chave da API Anthropic (opcional)
        gemini_key: Chave da API Google (opcional)
        force_method: Forçar método específico ('ocr', 'gemini', 'claude')

    Returns:
        Dicionário com resultado completo
    """
    result = ExtractionResult(
        dados_estruturados={},
        resumo="",
        metodo_usado="none",
        confianca=0.0
    )

    extracted_data = {}
    ocr_text = ""
    missing_fields = []

    # =========================================
    # CAMADA 1: OCR Local + Regex (Custo Zero)
    # =========================================
    if force_method in (None, 'ocr'):
        try:
            from ocr_local import extract_text_from_image, preprocess_image
            from regex_extractor import extract_with_regex

            # Pré-processar imagem para melhor OCR
            processed_image = preprocess_image(image_data)

            # Extrair texto com OCR
            ocr_text, ocr_backend = extract_text_from_image(processed_image)

            if ocr_text and len(ocr_text.strip()) > 50:
                result.ocr_backend = ocr_backend
                result.texto_ocr = ocr_text

                # Extrair com regex
                regex_result = extract_with_regex(ocr_text)
                extracted_data = _build_complete_structure(regex_result)
                missing_fields = regex_result.get('campos_faltantes', [])
                result.confianca = regex_result.get('confianca', 0.0)

                # Se confiança alta, usar apenas OCR+Regex
                if result.confianca >= 0.7:
                    result.metodo_usado = f"ocr_{ocr_backend}+regex"
                    result.dados_estruturados = extracted_data
                    result.tokens_estimados = 0
                    result.custo_estimado = 0.0

                    # Gerar resumo localmente se possível
                    result.resumo = _generate_local_summary(extracted_data)
                    return _result_to_dict(result)

        except ImportError as e:
            print(f"OCR local não disponível: {e}")
        except Exception as e:
            print(f"Erro no OCR local: {e}")

    # =========================================
    # CAMADA 2: Gemini com Texto OCR (Baixo Custo)
    # =========================================
    if gemini_key and missing_fields and ocr_text and force_method in (None, 'gemini'):
        try:
            from gemini_fallback import complete_with_gemini_text

            gemini_additions = complete_with_gemini_text(
                ocr_text, missing_fields, extracted_data, gemini_key
            )

            if gemini_additions:
                extracted_data = _merge_data(extracted_data, gemini_additions)
                result.metodo_usado = f"ocr_{result.ocr_backend}+regex+gemini_text"
                result.tokens_estimados = len(ocr_text) // 4
                result.custo_estimado = result.tokens_estimados * 0.000001  # ~$0.001/1M tokens

                # Recalcular campos faltantes
                missing_fields = _get_missing_fields(extracted_data)

                if len(missing_fields) <= 2:
                    result.dados_estruturados = extracted_data
                    result.confianca = 0.8
                    result.resumo = _generate_local_summary(extracted_data)
                    return _result_to_dict(result)

        except ImportError:
            print("Gemini não disponível")
        except Exception as e:
            print(f"Erro no Gemini text: {e}")

    # =========================================
    # CAMADA 3: Gemini Vision (Médio Custo)
    # =========================================
    if gemini_key and force_method in (None, 'gemini'):
        try:
            from gemini_fallback import extract_full_with_gemini, generate_summary_gemini

            if missing_fields and extracted_data:
                # Completar campos faltantes
                from gemini_fallback import complete_with_gemini_vision
                gemini_vision_data = complete_with_gemini_vision(
                    image_data, media_type, missing_fields, extracted_data, gemini_key
                )
                extracted_data = _merge_data(extracted_data, gemini_vision_data)
            else:
                # Extração completa
                extracted_data = extract_full_with_gemini(image_data, media_type, gemini_key)
                extracted_data = _build_complete_structure(extracted_data)

            result.metodo_usado = "gemini_vision"
            result.dados_estruturados = extracted_data
            result.confianca = 0.85
            result.tokens_estimados = 1000  # Estimativa para imagem
            result.custo_estimado = 0.001  # ~$0.001 por imagem

            # Gerar resumo com Gemini
            result.resumo = generate_summary_gemini(extracted_data, gemini_key)

            return _result_to_dict(result)

        except ImportError:
            print("Gemini não disponível")
        except Exception as e:
            print(f"Erro no Gemini vision: {e}")

    # =========================================
    # CAMADA 4: Claude Vision (Maior Precisão)
    # =========================================
    if anthropic_key and force_method in (None, 'claude'):
        try:
            extracted_data = _extract_with_claude(image_data, media_type, anthropic_key)
            extracted_data = _build_complete_structure(extracted_data)

            result.metodo_usado = "claude_vision"
            result.dados_estruturados = extracted_data
            result.confianca = 0.95
            result.tokens_estimados = 2000
            result.custo_estimado = 0.01  # ~$0.01 por imagem com Claude

            # Gerar resumo com Claude
            result.resumo = _generate_summary_claude(extracted_data, anthropic_key)

            return _result_to_dict(result)

        except Exception as e:
            print(f"Erro no Claude: {e}")

    # Se nenhum método funcionou
    result.metodo_usado = "fallback_local"
    result.dados_estruturados = extracted_data if extracted_data else _build_complete_structure({})
    result.resumo = _generate_local_summary(result.dados_estruturados) if extracted_data else "Não foi possível extrair dados do receituário."

    return _result_to_dict(result)


def _extract_with_claude(image_data: bytes, media_type: str, api_key: str) -> Dict[str, Any]:
    """Extrai dados usando Claude Vision."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    image_base64 = base64.standard_b64encode(image_data).decode("utf-8")

    extraction_prompt = """Analise esta imagem de um receituário/prescrição médica e extraia TODAS as informações disponíveis.

Retorne APENAS um JSON válido (sem markdown, sem ```json```, apenas o JSON puro) com a seguinte estrutura:

{
    "medico": {
        "nome": "nome completo do médico",
        "crm": "número do CRM",
        "especialidade": "especialidade médica",
        "email": "email se disponível",
        "telefones": ["lista de telefones"],
        "endereco": "endereço do consultório",
        "website": "website se disponível"
    },
    "paciente": {
        "nome": "nome do paciente"
    },
    "documento": {
        "tipo": "tipo do documento (receita, solicitação de exame, atestado, etc)",
        "data": "data do documento se disponível"
    },
    "prescricoes": [
        {
            "tipo": "medicamento ou exame",
            "descricao": "descrição completa do medicamento/exame",
            "posologia": "posologia se for medicamento",
            "quantidade": "quantidade se especificada"
        }
    ],
    "indicacao_clinica": "indicação clínica ou diagnóstico mencionado",
    "observacoes": "quaisquer observações adicionais importantes",
    "historico_relevante": "histórico médico relevante mencionado no documento"
}

Se algum campo não estiver disponível na imagem, use null.
Extraia todas as informações visíveis, mesmo que parcialmente legíveis.
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": extraction_prompt
                    }
                ]
            }
        ]
    )

    response_text = message.content[0].text

    try:
        json_text = response_text.strip()
        if json_text.startswith("```"):
            json_text = re.sub(r'^```json?\n?', '', json_text)
            json_text = re.sub(r'\n?```$', '', json_text)

        return json.loads(json_text)
    except json.JSONDecodeError:
        return {"erro": "Não foi possível estruturar os dados", "texto_extraido": response_text}


def _generate_summary_claude(prescription_data: Dict[str, Any], api_key: str) -> str:
    """Gera resumo usando Claude."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    summary_prompt = f"""Com base nos dados extraídos de um receituário médico, gere um resumo claro e conciso em português brasileiro.
O resumo deve ser informativo e fácil de entender para o paciente.

Dados do receituário:
{json.dumps(prescription_data, ensure_ascii=False, indent=2)}

Gere um resumo organizado com:
1. Informações do médico
2. Informações do paciente
3. O que foi prescrito/solicitado
4. Indicações clínicas (se houver)
5. Próximos passos recomendados

Seja objetivo e use linguagem acessível."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": summary_prompt
            }
        ]
    )

    return message.content[0].text


def _generate_local_summary(data: Dict[str, Any]) -> str:
    """Gera resumo simples localmente sem usar API."""
    parts = []

    # Médico
    medico = data.get('medico', {})
    if medico.get('nome'):
        med_info = f"**Médico:** {medico['nome']}"
        if medico.get('especialidade'):
            med_info += f" ({medico['especialidade']})"
        if medico.get('crm'):
            med_info += f" - CRM: {medico['crm']}"
        parts.append(med_info)

    # Paciente
    paciente = data.get('paciente', {})
    if paciente.get('nome'):
        parts.append(f"**Paciente:** {paciente['nome']}")

    # Documento
    documento = data.get('documento', {})
    if documento.get('tipo'):
        doc_info = f"**Tipo de documento:** {documento['tipo']}"
        if documento.get('data'):
            doc_info += f" - Data: {documento['data']}"
        parts.append(doc_info)

    # Prescrições
    prescricoes = data.get('prescricoes', [])
    if prescricoes:
        parts.append("**Prescrições/Exames:**")
        for i, p in enumerate(prescricoes, 1):
            desc = p.get('descricao', 'Não especificado')
            tipo = p.get('tipo', '').upper()
            parts.append(f"  {i}. [{tipo}] {desc}")

    # Indicação clínica
    if data.get('indicacao_clinica'):
        parts.append(f"**Indicação clínica:** {data['indicacao_clinica']}")

    # Histórico
    if data.get('historico_relevante'):
        parts.append(f"**Histórico relevante:** {data['historico_relevante']}")

    if not parts:
        return "Não foi possível extrair informações suficientes do receituário."

    return "\n\n".join(parts)


def _get_missing_fields(data: Dict[str, Any]) -> list:
    """Identifica campos principais que estão faltando."""
    missing = []

    if not data.get('medico', {}).get('nome'):
        missing.append('medico.nome')
    if not data.get('medico', {}).get('crm'):
        missing.append('medico.crm')
    if not data.get('paciente', {}).get('nome'):
        missing.append('paciente.nome')
    if not data.get('documento', {}).get('tipo'):
        missing.append('documento.tipo')
    if not data.get('prescricoes'):
        missing.append('prescricoes')
    if not data.get('indicacao_clinica'):
        missing.append('indicacao_clinica')

    return missing


def _result_to_dict(result: ExtractionResult) -> Dict[str, Any]:
    """Converte ExtractionResult para dicionário."""
    return {
        "dados_estruturados": result.dados_estruturados,
        "resumo": result.resumo,
        "metadados": {
            "metodo_usado": result.metodo_usado,
            "ocr_backend": result.ocr_backend,
            "confianca": result.confianca,
            "tokens_estimados": result.tokens_estimados,
            "custo_estimado_usd": result.custo_estimado
        },
        "texto_ocr": result.texto_ocr
    }


# Manter compatibilidade com versão anterior
def process_prescription(image_data: bytes, media_type: str, api_key: str) -> dict:
    """
    Processa completamente um receituário médico.
    Mantido para compatibilidade - usa apenas Claude.

    Args:
        image_data: Bytes da imagem
        media_type: Tipo MIME da imagem
        api_key: Chave da API da Anthropic

    Returns:
        Dicionário com dados estruturados e resumo
    """
    return process_prescription_optimized(
        image_data=image_data,
        media_type=media_type,
        anthropic_key=api_key,
        force_method='claude'
    )


def extract_prescription_data(image_data: bytes, media_type: str, api_key: str) -> dict:
    """Mantido para compatibilidade."""
    return _extract_with_claude(image_data, media_type, api_key)


def generate_summary(prescription_data: dict, api_key: str) -> str:
    """Mantido para compatibilidade."""
    return _generate_summary_claude(prescription_data, api_key)
