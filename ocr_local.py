"""
Módulo de OCR local para extração de texto de imagens.
Suporta múltiplos backends: docling, pytesseract, easyocr.
"""

import io
from typing import Optional, Tuple
from PIL import Image


class OCRProcessor:
    """Processador de OCR com fallback entre diferentes backends."""

    def __init__(self):
        self.available_backends = []
        self._detect_backends()

    def _detect_backends(self):
        """Detecta quais backends de OCR estão disponíveis."""
        # Tentar docling
        try:
            from docling.document_converter import DocumentConverter
            self.available_backends.append("docling")
        except ImportError:
            pass

        # Tentar pytesseract
        try:
            import pytesseract
            # Verificar se o tesseract está instalado
            pytesseract.get_tesseract_version()
            self.available_backends.append("pytesseract")
        except (ImportError, Exception):
            pass

        # Tentar easyocr
        try:
            import easyocr
            self.available_backends.append("easyocr")
        except ImportError:
            pass

    def extract_text(self, image_data: bytes, language: str = "por") -> Tuple[str, str]:
        """
        Extrai texto de uma imagem usando o backend disponível.

        Args:
            image_data: Bytes da imagem
            language: Código do idioma (por=português)

        Returns:
            Tupla (texto_extraido, backend_usado)
        """
        if not self.available_backends:
            return "", "none"

        # Tentar cada backend em ordem de preferência
        for backend in ["docling", "pytesseract", "easyocr"]:
            if backend in self.available_backends:
                try:
                    if backend == "docling":
                        text = self._extract_with_docling(image_data)
                    elif backend == "pytesseract":
                        text = self._extract_with_pytesseract(image_data, language)
                    elif backend == "easyocr":
                        text = self._extract_with_easyocr(image_data, language)

                    if text and len(text.strip()) > 50:  # Texto mínimo válido
                        return text, backend
                except Exception as e:
                    print(f"Erro com {backend}: {e}")
                    continue

        return "", "none"

    def _extract_with_docling(self, image_data: bytes) -> str:
        """Extrai texto usando docling."""
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        import tempfile
        import os

        # Salvar imagem temporariamente
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        try:
            converter = DocumentConverter()
            result = converter.convert(tmp_path)
            text = result.document.export_to_markdown()
            return text
        finally:
            os.unlink(tmp_path)

    def _extract_with_pytesseract(self, image_data: bytes, language: str) -> str:
        """Extrai texto usando pytesseract."""
        import pytesseract

        # Mapear código de idioma
        lang_map = {"por": "por", "eng": "eng", "pt": "por", "en": "eng"}
        tesseract_lang = lang_map.get(language, "por")

        image = Image.open(io.BytesIO(image_data))

        # Configurações para melhor extração
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(
            image,
            lang=tesseract_lang,
            config=custom_config
        )
        return text

    def _extract_with_easyocr(self, image_data: bytes, language: str) -> str:
        """Extrai texto usando easyocr."""
        import easyocr
        import numpy as np

        # Mapear código de idioma
        lang_map = {"por": "pt", "eng": "en", "pt": "pt", "en": "en"}
        easyocr_lang = lang_map.get(language, "pt")

        # Criar reader (com cache)
        if not hasattr(self, '_easyocr_reader'):
            self._easyocr_reader = easyocr.Reader([easyocr_lang, 'en'], gpu=False)

        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)

        results = self._easyocr_reader.readtext(image_np)

        # Concatenar textos extraídos
        text = "\n".join([result[1] for result in results])
        return text


def preprocess_image(image_data: bytes) -> bytes:
    """
    Pré-processa a imagem para melhorar a qualidade do OCR.

    Args:
        image_data: Bytes da imagem original

    Returns:
        Bytes da imagem processada
    """
    try:
        from PIL import ImageEnhance, ImageFilter

        image = Image.open(io.BytesIO(image_data))

        # Converter para RGB se necessário
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Aumentar contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # Aumentar nitidez
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)

        # Converter para escala de cinza para OCR
        image = image.convert('L')

        # Binarização simples
        threshold = 128
        image = image.point(lambda p: 255 if p > threshold else 0)

        # Salvar em bytes
        output = io.BytesIO()
        image.save(output, format='PNG')
        return output.getvalue()

    except Exception:
        return image_data


# Instância global do processador
_ocr_processor: Optional[OCRProcessor] = None


def get_ocr_processor() -> OCRProcessor:
    """Retorna a instância global do processador OCR."""
    global _ocr_processor
    if _ocr_processor is None:
        _ocr_processor = OCRProcessor()
    return _ocr_processor


def extract_text_from_image(image_data: bytes, language: str = "por") -> Tuple[str, str]:
    """
    Função de conveniência para extrair texto de uma imagem.

    Args:
        image_data: Bytes da imagem
        language: Código do idioma

    Returns:
        Tupla (texto_extraido, backend_usado)
    """
    processor = get_ocr_processor()
    return processor.extract_text(image_data, language)
