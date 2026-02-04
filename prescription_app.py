"""
Aplicação Streamlit para extração de informações de receituários médicos.
Versão otimizada com estratégia em camadas para economia de tokens.
"""

import streamlit as st
import json
from extractor import process_prescription_optimized

# Configuração da página
st.set_page_config(
    page_title="Leitor de Receituário Médico",
    page_icon="🏥",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .success-card {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
    }
    .warning-card {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
    }
    .cost-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
    }
    .method-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .method-ocr {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    .method-gemini {
        background-color: #bbdefb;
        color: #1565c0;
    }
    .method-claude {
        background-color: #e1bee7;
        color: #7b1fa2;
    }
</style>
""", unsafe_allow_html=True)


def get_method_badge(method: str) -> str:
    """Retorna o badge HTML para o método usado."""
    if 'ocr' in method.lower() and 'gemini' not in method.lower() and 'claude' not in method.lower():
        return f'<span class="method-badge method-ocr">🟢 {method}</span>'
    elif 'gemini' in method.lower():
        return f'<span class="method-badge method-gemini">🔵 {method}</span>'
    elif 'claude' in method.lower():
        return f'<span class="method-badge method-claude">🟣 {method}</span>'
    else:
        return f'<span class="method-badge">{method}</span>'


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Leitor de Receituário Médico</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extraia informações estruturadas de prescrições médicas com IA - Otimizado para economia de tokens</p>', unsafe_allow_html=True)

    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")

        st.subheader("🔑 Chaves de API")

        anthropic_key = st.text_input(
            "Chave Anthropic (Claude)",
            type="password",
            help="Maior precisão, maior custo (~$0.01/imagem)"
        )

        gemini_key = st.text_input(
            "Chave Google (Gemini)",
            type="password",
            help="Boa precisão, baixo custo (~$0.001/imagem)"
        )

        st.markdown("---")

        st.subheader("🎯 Estratégia de Extração")

        extraction_mode = st.radio(
            "Modo de extração",
            options=[
                "auto",
                "economia",
                "precisao",
                "apenas_ocr"
            ],
            format_func=lambda x: {
                "auto": "🔄 Automático (recomendado)",
                "economia": "💰 Economia máxima (Gemini)",
                "precisao": "🎯 Máxima precisão (Claude)",
                "apenas_ocr": "🆓 Apenas OCR local (grátis)"
            }[x],
            help="Escolha a estratégia de extração baseada em custo vs. precisão"
        )

        st.markdown("---")

        st.markdown("""
        ### 📊 Comparação de Custos

        | Método | Custo | Precisão |
        |--------|-------|----------|
        | OCR+Regex | $0.00 | ⭐⭐⭐ |
        | Gemini | ~$0.001 | ⭐⭐⭐⭐ |
        | Claude | ~$0.01 | ⭐⭐⭐⭐⭐ |

        ### 📋 Como funciona:
        1. **Camada 1**: OCR local extrai texto
        2. **Camada 2**: Regex identifica campos
        3. **Camada 3**: Gemini completa falhas
        4. **Camada 4**: Claude como fallback final
        """)

        st.markdown("---")
        st.markdown("### 📄 Formatos aceitos:")
        st.markdown("JPG, PNG, GIF, WebP")

    # Área principal
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload do Receituário")

        uploaded_file = st.file_uploader(
            "Selecione a imagem do receituário",
            type=["jpg", "jpeg", "png", "gif", "webp"],
            help="Faça upload de uma imagem clara do receituário médico"
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Imagem carregada", use_container_width=True)

            # Informações do arquivo
            file_details = {
                "Nome": uploaded_file.name,
                "Tipo": uploaded_file.type,
                "Tamanho": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.json(file_details)

    with col2:
        st.subheader("🔍 Processamento")

        # Verificar se tem pelo menos uma forma de processar
        can_process = uploaded_file is not None

        if extraction_mode == "precisao" and not anthropic_key:
            st.warning("⚠️ Modo precisão requer chave Anthropic.")
            can_process = False
        elif extraction_mode == "economia" and not gemini_key:
            st.warning("⚠️ Modo economia requer chave Gemini.")
            can_process = False
        elif extraction_mode == "auto" and not (anthropic_key or gemini_key):
            st.info("💡 Modo automático funcionará apenas com OCR local. Adicione uma chave de API para melhor precisão.")

        if uploaded_file is None:
            st.info("📤 Faça upload de uma imagem para começar.")

        if can_process:
            if st.button("🚀 Processar Receituário", type="primary", use_container_width=True):
                with st.spinner("Analisando receituário..."):
                    try:
                        # Ler dados da imagem
                        image_data = uploaded_file.getvalue()
                        media_type = uploaded_file.type

                        # Determinar método forçado baseado no modo
                        force_method = None
                        if extraction_mode == "economia":
                            force_method = "gemini"
                        elif extraction_mode == "precisao":
                            force_method = "claude"
                        elif extraction_mode == "apenas_ocr":
                            force_method = "ocr"

                        # Processar receituário
                        result = process_prescription_optimized(
                            image_data=image_data,
                            media_type=media_type,
                            anthropic_key=anthropic_key if anthropic_key else None,
                            gemini_key=gemini_key if gemini_key else None,
                            force_method=force_method
                        )

                        # Armazenar resultado na sessão
                        st.session_state['result'] = result
                        st.session_state['processed'] = True

                        st.success("✅ Processamento concluído!")

                    except Exception as e:
                        st.error(f"❌ Erro ao processar: {str(e)}")

    # Exibir resultados
    if 'processed' in st.session_state and st.session_state.get('processed'):
        st.markdown("---")

        result = st.session_state['result']
        metadados = result.get('metadados', {})

        # Card de informações do processamento
        st.markdown("### 📊 Informações do Processamento")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            method = metadados.get('metodo_usado', 'N/A')
            st.markdown(f"**Método:** {get_method_badge(method)}", unsafe_allow_html=True)

        with col_m2:
            confianca = metadados.get('confianca', 0) * 100
            st.metric("Confiança", f"{confianca:.0f}%")

        with col_m3:
            tokens = metadados.get('tokens_estimados', 0)
            st.metric("Tokens usados", f"{tokens:,}")

        with col_m4:
            custo = metadados.get('custo_estimado_usd', 0)
            st.metric("Custo estimado", f"${custo:.4f}")

        # Tabs para diferentes visualizações
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dados Estruturados", "📝 Resumo", "🔤 Texto OCR", "💾 JSON Completo"])

        with tab1:
            st.subheader("Dados Extraídos do Receituário")

            dados = result.get('dados_estruturados', {})

            # Informações do Médico
            if dados.get('medico'):
                st.markdown("#### 👨‍⚕️ Informações do Médico")
                medico = dados['medico']

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    if medico.get('nome'):
                        st.markdown(f"**Nome:** {medico['nome']}")
                    if medico.get('crm'):
                        st.markdown(f"**CRM:** {medico['crm']}")
                    if medico.get('especialidade'):
                        st.markdown(f"**Especialidade:** {medico['especialidade']}")

                with col_m2:
                    if medico.get('email'):
                        st.markdown(f"**Email:** {medico['email']}")
                    if medico.get('website'):
                        st.markdown(f"**Website:** {medico['website']}")
                    if medico.get('telefones'):
                        phones = medico['telefones']
                        if isinstance(phones, list):
                            st.markdown(f"**Telefones:** {', '.join(phones)}")
                        else:
                            st.markdown(f"**Telefones:** {phones}")

                if medico.get('endereco'):
                    st.markdown(f"**Endereço:** {medico['endereco']}")

                st.markdown("---")

            # Informações do Paciente
            if dados.get('paciente'):
                st.markdown("#### 🧑 Informações do Paciente")
                paciente = dados['paciente']
                if paciente.get('nome'):
                    st.markdown(f"**Nome:** {paciente['nome']}")
                st.markdown("---")

            # Tipo de Documento
            if dados.get('documento'):
                st.markdown("#### 📄 Tipo de Documento")
                documento = dados['documento']
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if documento.get('tipo'):
                        st.markdown(f"**Tipo:** {documento['tipo']}")
                with col_d2:
                    if documento.get('data'):
                        st.markdown(f"**Data:** {documento['data']}")
                st.markdown("---")

            # Prescrições/Exames
            if dados.get('prescricoes'):
                st.markdown("#### 💊 Prescrições/Exames Solicitados")
                for i, prescricao in enumerate(dados['prescricoes'], 1):
                    with st.expander(f"Item {i}: {prescricao.get('tipo', 'N/A').upper()}", expanded=True):
                        st.markdown(f"**Descrição:** {prescricao.get('descricao', 'N/A')}")
                        if prescricao.get('posologia'):
                            st.markdown(f"**Posologia:** {prescricao['posologia']}")
                        if prescricao.get('quantidade'):
                            st.markdown(f"**Quantidade:** {prescricao['quantidade']}")
                st.markdown("---")

            # Indicação Clínica
            if dados.get('indicacao_clinica'):
                st.markdown("#### 🩺 Indicação Clínica")
                st.markdown(f"<div class='info-card'>{dados['indicacao_clinica']}</div>", unsafe_allow_html=True)
                st.markdown("---")

            # Histórico Relevante
            if dados.get('historico_relevante'):
                st.markdown("#### 📋 Histórico Médico Relevante")
                st.markdown(f"<div class='warning-card'>{dados['historico_relevante']}</div>", unsafe_allow_html=True)
                st.markdown("---")

            # Observações
            if dados.get('observacoes'):
                st.markdown("#### 📌 Observações")
                st.info(dados['observacoes'])

        with tab2:
            st.subheader("Resumo do Receituário")
            resumo = result.get('resumo', 'Resumo não disponível')
            st.markdown(f"<div class='success-card'>{resumo}</div>", unsafe_allow_html=True)

        with tab3:
            st.subheader("Texto Extraído pelo OCR")
            texto_ocr = result.get('texto_ocr')
            if texto_ocr:
                ocr_backend = metadados.get('ocr_backend', 'desconhecido')
                st.caption(f"Extraído com: {ocr_backend}")
                st.code(texto_ocr, language=None)
            else:
                st.info("OCR local não foi utilizado ou não está disponível.")

        with tab4:
            st.subheader("JSON Completo")

            # Botão para copiar JSON
            json_str = json.dumps(result, ensure_ascii=False, indent=2)

            st.download_button(
                label="📥 Baixar JSON",
                data=json_str,
                file_name="receituario_extraido.json",
                mime="application/json"
            )

            st.code(json_str, language="json")

        # Botão para limpar e processar novo
        if st.button("🔄 Processar Novo Receituário"):
            st.session_state['processed'] = False
            st.session_state['result'] = None
            st.rerun()


if __name__ == "__main__":
    main()
