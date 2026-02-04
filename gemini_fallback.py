"""
Módulo de fallback usando Google Gemini API.
Usado quando o OCR local + regex não consegue extrair todos os campos.
"""

import base64
import json
import re
from typing import Dict, Any, List, Optional


def complete_with_gemini_text(
    text: str,
    missing_fields: List[str],
    partial_data: Dict[str, Any],
    api_key: str
) -> Dict[str, Any]:
    """
    Completa campos faltantes usando Gemini com texto extraído pelo OCR.
    Mais econômico que enviar a imagem.

    Args:
        text: Texto extraído pelo OCR
        missing_fields: Lista de campos que não foram extraídos
        partial_data: Dados parciais já extraídos
        api_key: Chave da API do Google

    Returns:
        Dicionário com os campos completados
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""Analise o texto abaixo extraído de um receituário médico e extraia APENAS os seguintes campos que estão faltando:
{', '.join(missing_fields)}

Texto do receituário:
---
{text}
---

Dados já extraídos (para contexto):
{json.dumps(partial_data, ensure_ascii=False, indent=2)}

Retorne APENAS um JSON válido com os campos solicitados.
Se não conseguir identificar algum campo, use null.
Não inclua explicações, apenas o JSON.

Estrutura esperada para cada campo:
- medico.nome: string com nome completo
- medico.crm: string com número do CRM
- medico.especialidade: string com especialidade médica
- medico.telefones: lista de strings com telefones
- medico.email: string com email
- medico.endereco: string com endereço
- medico.website: string com website
- paciente.nome: string com nome do paciente
- documento.tipo: string (receita, solicitação de exame, atestado, etc)
- documento.data: string com data
- prescricoes: lista de objetos com tipo, descricao, posologia, quantidade
- indicacao_clinica: string com indicação/diagnóstico
"""

    response = model.generate_content(prompt)
    response_text = response.text.strip()

    # Limpar possíveis marcadores markdown
    if response_text.startswith("```"):
        response_text = re.sub(r'^```json?\n?', '', response_text)
        response_text = re.sub(r'\n?```$', '', response_text)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {}


def complete_with_gemini_vision(
    image_data: bytes,
    media_type: str,
    missing_fields: List[str],
    partial_data: Dict[str, Any],
    api_key: str
) -> Dict[str, Any]:
    """
    Completa campos faltantes usando Gemini Vision com a imagem.
    Usado como último recurso quando o texto do OCR não é suficiente.

    Args:
        image_data: Bytes da imagem
        media_type: Tipo MIME da imagem
        missing_fields: Lista de campos faltantes
        partial_data: Dados parciais já extraídos
        api_key: Chave da API do Google

    Returns:
        Dicionário com os campos completados
    """
    import google.generativeai as genai
    from PIL import Image
    import io

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Converter bytes para PIL Image
    image = Image.open(io.BytesIO(image_data))

    prompt = f"""Analise esta imagem de um receituário médico e extraia APENAS os seguintes campos:
{', '.join(missing_fields)}

Dados já extraídos (para contexto, não repetir):
{json.dumps(partial_data, ensure_ascii=False, indent=2)}

Retorne APENAS um JSON válido com os campos solicitados.
Se não conseguir identificar algum campo, use null.
Não inclua explicações, apenas o JSON.
"""

    response = model.generate_content([prompt, image])
    response_text = response.text.strip()

    # Limpar possíveis marcadores markdown
    if response_text.startswith("```"):
        response_text = re.sub(r'^```json?\n?', '', response_text)
        response_text = re.sub(r'\n?```$', '', response_text)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {}


def extract_full_with_gemini(
    image_data: bytes,
    media_type: str,
    api_key: str
) -> Dict[str, Any]:
    """
    Extração completa usando Gemini Vision.
    Usado quando não há OCR local disponível.

    Args:
        image_data: Bytes da imagem
        media_type: Tipo MIME da imagem
        api_key: Chave da API do Google

    Returns:
        Dicionário com todos os dados extraídos
    """
    import google.generativeai as genai
    from PIL import Image
    import io

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    image = Image.open(io.BytesIO(image_data))

    prompt = """Analise esta imagem de um receituário/prescrição médica e extraia TODAS as informações disponíveis.

Retorne APENAS um JSON válido com a seguinte estrutura:

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
        "tipo": "tipo do documento",
        "data": "data do documento"
    },
    "prescricoes": [
        {
            "tipo": "medicamento ou exame",
            "descricao": "descrição completa",
            "posologia": "posologia se medicamento",
            "quantidade": "quantidade"
        }
    ],
    "indicacao_clinica": "indicação clínica ou diagnóstico",
    "observacoes": "observações adicionais",
    "historico_relevante": "histórico médico relevante"
}

Se algum campo não estiver disponível, use null.
Não inclua explicações, apenas o JSON.
"""

    response = model.generate_content([prompt, image])
    response_text = response.text.strip()

    # Limpar possíveis marcadores markdown
    if response_text.startswith("```"):
        response_text = re.sub(r'^```json?\n?', '', response_text)
        response_text = re.sub(r'\n?```$', '', response_text)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"erro": "Falha ao processar resposta", "texto_bruto": response_text}


def generate_summary_gemini(prescription_data: Dict[str, Any], api_key: str) -> str:
    """
    Gera um resumo usando Gemini (mais econômico que Claude para texto).

    Args:
        prescription_data: Dados do receituário
        api_key: Chave da API do Google

    Returns:
        Resumo em português
    """
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""Com base nos dados extraídos de um receituário médico, gere um resumo claro e conciso em português brasileiro.

Dados do receituário:
{json.dumps(prescription_data, ensure_ascii=False, indent=2)}

Gere um resumo organizado com:
1. Informações do médico
2. Informações do paciente
3. O que foi prescrito/solicitado
4. Indicações clínicas (se houver)
5. Próximos passos recomendados

Seja objetivo e use linguagem acessível."""

    response = model.generate_content(prompt)
    return response.text
