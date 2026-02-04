"""
Módulo de extração de informações de receituários médicos.
Usa a API da Anthropic (Claude) para processar imagens e extrair dados estruturados.
"""

import anthropic
import base64
import json
import re
from pathlib import Path


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


def extract_prescription_data(image_data: bytes, media_type: str, api_key: str) -> dict:
    """
    Extrai informações estruturadas de um receituário médico usando Claude.

    Args:
        image_data: Bytes da imagem
        media_type: Tipo MIME da imagem (ex: image/jpeg)
        api_key: Chave da API da Anthropic

    Returns:
        Dicionário com os dados estruturados do receituário
    """
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

    # Tentar extrair JSON da resposta
    try:
        # Remover possíveis marcadores de código markdown
        json_text = response_text.strip()
        if json_text.startswith("```"):
            json_text = re.sub(r'^```json?\n?', '', json_text)
            json_text = re.sub(r'\n?```$', '', json_text)

        prescription_data = json.loads(json_text)
    except json.JSONDecodeError:
        # Se falhar, retornar o texto bruto em um formato estruturado
        prescription_data = {
            "erro": "Não foi possível estruturar os dados",
            "texto_extraido": response_text
        }

    return prescription_data


def generate_summary(prescription_data: dict, api_key: str) -> str:
    """
    Gera um resumo em linguagem natural dos dados extraídos.

    Args:
        prescription_data: Dicionário com os dados do receituário
        api_key: Chave da API da Anthropic

    Returns:
        String com o resumo em português
    """
    client = anthropic.Anthropic(api_key=api_key)

    summary_prompt = f"""Com base nos dados extraídos de um receituário médico abaixo, gere um resumo claro e conciso em português brasileiro.
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


def process_prescription(image_data: bytes, media_type: str, api_key: str) -> dict:
    """
    Processa completamente um receituário médico.

    Args:
        image_data: Bytes da imagem
        media_type: Tipo MIME da imagem
        api_key: Chave da API da Anthropic

    Returns:
        Dicionário com dados estruturados e resumo
    """
    # Extrair dados estruturados
    prescription_data = extract_prescription_data(image_data, media_type, api_key)

    # Gerar resumo se a extração foi bem sucedida
    if "erro" not in prescription_data:
        summary = generate_summary(prescription_data, api_key)
    else:
        summary = "Não foi possível gerar um resumo devido a erros na extração."

    return {
        "dados_estruturados": prescription_data,
        "resumo": summary
    }
