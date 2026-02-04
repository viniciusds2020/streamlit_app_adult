"""
Módulo de extração de dados usando expressões regulares.
Otimizado para receituários médicos brasileiros.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ExtractedData:
    """Estrutura para dados extraídos."""
    medico: Dict[str, Any] = field(default_factory=dict)
    paciente: Dict[str, Any] = field(default_factory=dict)
    documento: Dict[str, Any] = field(default_factory=dict)
    prescricoes: List[Dict[str, Any]] = field(default_factory=list)
    indicacao_clinica: Optional[str] = None
    observacoes: Optional[str] = None
    historico_relevante: Optional[str] = None
    campos_extraidos: List[str] = field(default_factory=list)
    campos_faltantes: List[str] = field(default_factory=list)
    confianca: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RegexExtractor:
    """Extrator de informações usando regex."""

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Compila os padrões regex para melhor performance."""

        # Padrões para CRM (diferentes formatos)
        self.crm_patterns = [
            re.compile(r'CRM[:\s-]*(\d{1,2}[\.\s]?\d{3,6}(?:[/-]\d{1,2})?)', re.IGNORECASE),
            re.compile(r'CRM[:\s-]*([A-Z]{2})[:\s-]*(\d{4,6})', re.IGNORECASE),
            re.compile(r'(\d{4,6})[/-]?([A-Z]{2})\s*CRM', re.IGNORECASE),
        ]

        # Padrões para nome do médico
        self.doctor_patterns = [
            re.compile(r'(?:Dr\.?|Dra\.?|Doutor|Doutora)\s+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+){1,5})', re.IGNORECASE),
            re.compile(r'^([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+){1,5})\s*\n.*?CRM', re.MULTILINE | re.IGNORECASE),
        ]

        # Padrões para especialidade médica
        self.specialty_patterns = [
            re.compile(r'(Endocrinolog(?:ia|ista)|Cardiolog(?:ia|ista)|Dermatolog(?:ia|ista)|'
                       r'Ginecolog(?:ia|ista)|Neurolog(?:ia|ista)|Ortoped(?:ia|ista)|'
                       r'Pediatr(?:ia|a)|Psiquiatr(?:ia|a)|Urolog(?:ia|ista)|'
                       r'Oftalmolog(?:ia|ista)|Otorrinolaringolog(?:ia|ista)|'
                       r'Gastroenterolog(?:ia|ista)|Pneumolog(?:ia|ista)|'
                       r'Nefolog(?:ia|ista)|Hematolog(?:ia|ista)|Oncolog(?:ia|ista)|'
                       r'Reumatolog(?:ia|ista)|Infectolog(?:ia|ista)|Geriatr(?:ia|a)|'
                       r'Clínic[oa]\s+Geral|Medicina\s+(?:Interna|do\s+Trabalho|Esportiva))',
                       re.IGNORECASE),
        ]

        # Padrões para telefone
        self.phone_patterns = [
            re.compile(r'(?:Tel(?:efone)?|Cel(?:ular)?|Fone)[:\s]*\(?(\d{2})\)?[\s.-]*(\d{4,5})[\s.-]*(\d{4})', re.IGNORECASE),
            re.compile(r'\(?(\d{2})\)?[\s.-]*(\d{4,5})[\s.-]*(\d{4})'),
            re.compile(r'(\d{4,5})[\s.-]*(\d{4})'),
        ]

        # Padrões para email
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

        # Padrões para website
        self.website_pattern = re.compile(r'(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(?:/[^\s]*)?')

        # Padrões para endereço
        self.address_patterns = [
            re.compile(r'(?:Av\.?|Avenida|R\.?|Rua|Alameda|Al\.?|Travessa|Trav\.?|Praça|Pça\.?)\s+[^\n]+?(?:n[°º]?\.?\s*\d+)?[^\n]*?(?:sala|sl\.?|andar|apto?\.?|apartamento)?\s*\d*', re.IGNORECASE),
            re.compile(r'(?:Consultório|Clínica)[:\s]+([^\n]+)', re.IGNORECASE),
        ]

        # Padrões para nome do paciente
        self.patient_patterns = [
            re.compile(r'(?:Paciente|Nome|Para)[:\s]+([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+){1,5})', re.IGNORECASE),
            re.compile(r'(?:Solicita[çc][ãa]o\s+de\s+(?:Exames?|Servi[çc]os?)\s+(?:e\s+Servi[çc]os?)?\s*\n+)([A-ZÀ-Ú][a-zà-ú]+(?:\s+[A-ZÀ-Ú][a-zà-ú]+){1,5})', re.IGNORECASE | re.MULTILINE),
        ]

        # Padrões para data
        self.date_patterns = [
            re.compile(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})'),
            re.compile(r'(\d{1,2})\s+de\s+([a-zç]+)\s+de\s+(\d{4})', re.IGNORECASE),
        ]

        # Padrões para tipo de documento
        self.doc_type_patterns = [
            re.compile(r'(Receitu[áa]rio|Prescri[çc][ãa]o|Solicita[çc][ãa]o\s+de\s+Exames?|'
                       r'Atestado|Laudo|Relat[óo]rio|Encaminhamento|Pedido\s+M[ée]dico)', re.IGNORECASE),
        ]

        # Padrões para exames comuns
        self.exam_patterns = [
            re.compile(r'(Ultrassonografia|USG|Ecografia|Tomografia|TC|Resson[âa]ncia|RM|RNM|'
                       r'Raio[s]?[\s-]?X|RX|Mamografia|Densitometria|Cintilografia|'
                       r'Eletrocardiograma|ECG|Ecocardiograma|Holter|MAPA|'
                       r'Endoscopia|Colonoscopia|Broncoscopia|'
                       r'Hemograma|Glicemia|TSH|T3|T4|Colesterol|Triglicer[íi]deos|'
                       r'Creatinina|Ureia|TGO|TGP|Bilirrubina|'
                       r'PSA|CEA|CA[\s-]?\d+|AFP|Beta[\s-]?HCG|'
                       r'Urina|Fezes|Cultura|Antibiograma)[^\n]*',
                       re.IGNORECASE),
        ]

        # Padrões para medicamentos comuns (genéricos e comerciais)
        self.medication_patterns = [
            re.compile(r'(\d+)\s*(?:comp(?:rimido)?s?|c[áa]ps?(?:ula)?s?|ml|mg|g|gotas?|ampolas?|'
                       r'frascos?|sachês?|adesivos?|supositórios?)\s+(?:de\s+)?([^\n]+)', re.IGNORECASE),
            re.compile(r'([A-Za-zÀ-ú]+(?:\s+\d+\s*mg)?)\s*[-–]\s*(\d+)\s*(?:comp|cáps|cp)', re.IGNORECASE),
        ]

        # Padrões para indicação clínica
        self.indication_patterns = [
            re.compile(r'(?:Indica[çc][ãa]o|Diagn[óo]stico|Hip[óo]tese|CID|Motivo)[:\s=]+([^\n]+(?:\n(?![A-Z]{2,})[^\n]+)*)', re.IGNORECASE),
            re.compile(r'(?:CA|Carcinoma|C[âa]ncer|Tumor|Neoplasia)\s+[^\n]+', re.IGNORECASE),
        ]

        # Padrões para posologia
        self.posology_patterns = [
            re.compile(r'(\d+)\s*(?:vez(?:es)?|x)\s*(?:ao|por)\s*dia', re.IGNORECASE),
            re.compile(r'(?:de|a\s+cada)\s*(\d+)\s*(?:em\s*\d+\s*)?horas?', re.IGNORECASE),
            re.compile(r'(?:pela\s+)?manh[ãa]|(?:[àa]\s+)?(?:noite|tarde)|antes|ap[óo]s\s+(?:as\s+)?refei[çc][õo]es?|em\s+jejum', re.IGNORECASE),
        ]

    def extract(self, text: str) -> ExtractedData:
        """
        Extrai todas as informações possíveis do texto.

        Args:
            text: Texto extraído pelo OCR

        Returns:
            ExtractedData com os campos extraídos
        """
        data = ExtractedData()

        # Extrair informações do médico
        data.medico = self._extract_doctor_info(text)
        if data.medico.get('nome'):
            data.campos_extraidos.append('medico.nome')
        if data.medico.get('crm'):
            data.campos_extraidos.append('medico.crm')
        if data.medico.get('especialidade'):
            data.campos_extraidos.append('medico.especialidade')
        if data.medico.get('telefones'):
            data.campos_extraidos.append('medico.telefones')
        if data.medico.get('email'):
            data.campos_extraidos.append('medico.email')
        if data.medico.get('endereco'):
            data.campos_extraidos.append('medico.endereco')
        if data.medico.get('website'):
            data.campos_extraidos.append('medico.website')

        # Extrair informações do paciente
        data.paciente = self._extract_patient_info(text)
        if data.paciente.get('nome'):
            data.campos_extraidos.append('paciente.nome')

        # Extrair informações do documento
        data.documento = self._extract_document_info(text)
        if data.documento.get('tipo'):
            data.campos_extraidos.append('documento.tipo')
        if data.documento.get('data'):
            data.campos_extraidos.append('documento.data')

        # Extrair prescrições
        data.prescricoes = self._extract_prescriptions(text)
        if data.prescricoes:
            data.campos_extraidos.append('prescricoes')

        # Extrair indicação clínica
        data.indicacao_clinica = self._extract_indication(text)
        if data.indicacao_clinica:
            data.campos_extraidos.append('indicacao_clinica')

        # Determinar campos faltantes
        all_fields = [
            'medico.nome', 'medico.crm', 'medico.especialidade',
            'paciente.nome', 'documento.tipo', 'prescricoes', 'indicacao_clinica'
        ]
        data.campos_faltantes = [f for f in all_fields if f not in data.campos_extraidos]

        # Calcular confiança
        data.confianca = len(data.campos_extraidos) / len(all_fields)

        return data

    def _extract_doctor_info(self, text: str) -> Dict[str, Any]:
        """Extrai informações do médico."""
        info = {}

        # CRM
        for pattern in self.crm_patterns:
            match = pattern.search(text)
            if match:
                crm = ''.join(match.groups())
                info['crm'] = crm.strip()
                break

        # Nome do médico
        for pattern in self.doctor_patterns:
            match = pattern.search(text)
            if match:
                info['nome'] = match.group(1).strip()
                break

        # Especialidade
        for pattern in self.specialty_patterns:
            match = pattern.search(text)
            if match:
                info['especialidade'] = match.group(1).strip()
                break

        # Telefones
        phones = []
        for pattern in self.phone_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    phone = '-'.join(match)
                else:
                    phone = match
                phones.append(phone)
        if phones:
            info['telefones'] = list(set(phones))[:5]  # Máximo 5 telefones únicos

        # Email
        match = self.email_pattern.search(text)
        if match:
            info['email'] = match.group(0)

        # Website
        match = self.website_pattern.search(text)
        if match:
            website = match.group(0)
            if '@' not in website:  # Não confundir com email
                info['website'] = website

        # Endereço
        for pattern in self.address_patterns:
            match = pattern.search(text)
            if match:
                addr = match.group(0) if match.lastindex is None else match.group(1)
                info['endereco'] = addr.strip()
                break

        return info

    def _extract_patient_info(self, text: str) -> Dict[str, Any]:
        """Extrai informações do paciente."""
        info = {}

        for pattern in self.patient_patterns:
            match = pattern.search(text)
            if match:
                name = match.group(1).strip()
                # Verificar se não é um nome de médico ou título
                if not any(title in name.lower() for title in ['dr', 'dra', 'doutor', 'doutora']):
                    info['nome'] = name
                    break

        return info

    def _extract_document_info(self, text: str) -> Dict[str, Any]:
        """Extrai informações do documento."""
        info = {}

        # Tipo de documento
        for pattern in self.doc_type_patterns:
            match = pattern.search(text)
            if match:
                info['tipo'] = match.group(1).strip()
                break

        # Data
        for pattern in self.date_patterns:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    if groups[1].isdigit():
                        info['data'] = f"{groups[0]}/{groups[1]}/{groups[2]}"
                    else:
                        info['data'] = f"{groups[0]} de {groups[1]} de {groups[2]}"
                break

        return info

    def _extract_prescriptions(self, text: str) -> List[Dict[str, Any]]:
        """Extrai prescrições (exames ou medicamentos)."""
        prescriptions = []

        # Extrair exames
        for pattern in self.exam_patterns:
            matches = pattern.findall(text)
            for match in matches:
                exam_text = match if isinstance(match, str) else ' '.join(match)
                prescriptions.append({
                    'tipo': 'exame',
                    'descricao': exam_text.strip(),
                    'posologia': None,
                    'quantidade': None
                })

        # Extrair medicamentos
        for pattern in self.medication_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    prescriptions.append({
                        'tipo': 'medicamento',
                        'descricao': match[1].strip() if len(match) > 1 else match[0],
                        'posologia': None,
                        'quantidade': match[0] if match[0].isdigit() else None
                    })

        # Remover duplicatas
        seen = set()
        unique_prescriptions = []
        for p in prescriptions:
            key = p['descricao'].lower()[:50]
            if key not in seen:
                seen.add(key)
                unique_prescriptions.append(p)

        return unique_prescriptions

    def _extract_indication(self, text: str) -> Optional[str]:
        """Extrai indicação clínica."""
        for pattern in self.indication_patterns:
            match = pattern.search(text)
            if match:
                indication = match.group(0) if match.lastindex is None else match.group(1)
                return indication.strip()

        return None


# Instância global
_extractor: Optional[RegexExtractor] = None


def get_regex_extractor() -> RegexExtractor:
    """Retorna a instância global do extrator regex."""
    global _extractor
    if _extractor is None:
        _extractor = RegexExtractor()
    return _extractor


def extract_with_regex(text: str) -> Dict[str, Any]:
    """
    Função de conveniência para extrair dados usando regex.

    Args:
        text: Texto para extrair

    Returns:
        Dicionário com dados extraídos
    """
    extractor = get_regex_extractor()
    data = extractor.extract(text)
    return data.to_dict()
