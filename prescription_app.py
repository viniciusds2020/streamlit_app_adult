"""
Aplicação Streamlit para extração de informações de receituários médicos.
"""

import streamlit as st
import json
from extractor import process_prescription, extract_prescription_data, generate_summary

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
    .json-container {
        background-color: #263238;
        color: #fff;
        padding: 1rem;
        border-radius: 8px;
        overflow-x: auto;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Leitor de Receituário Médico</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extraia informações estruturadas de prescrições médicas usando Inteligência Artificial</p>', unsafe_allow_html=True)

    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")

        api_key = st.text_input(
            "Chave da API Anthropic",
            type="password",
            help="Insira sua chave da API da Anthropic para processar as imagens"
        )

        st.markdown("---")

        st.markdown("""
        ### 📋 Como usar:
        1. Insira sua chave da API
        2. Faça upload da imagem do receituário
        3. Clique em "Processar Receituário"
        4. Visualize os dados extraídos e o resumo

        ### 📄 Formatos aceitos:
        - JPG/JPEG
        - PNG
        - GIF
        - WebP
        """)

        st.markdown("---")
        st.markdown("### 🔒 Privacidade")
        st.info("Suas imagens são processadas de forma segura e não são armazenadas após o processamento.")

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

        if uploaded_file is not None and api_key:
            if st.button("🚀 Processar Receituário", type="primary", use_container_width=True):
                with st.spinner("Analisando receituário com IA..."):
                    try:
                        # Ler dados da imagem
                        image_data = uploaded_file.getvalue()
                        media_type = uploaded_file.type

                        # Processar receituário
                        result = process_prescription(image_data, media_type, api_key)

                        # Armazenar resultado na sessão
                        st.session_state['result'] = result
                        st.session_state['processed'] = True

                        st.success("✅ Processamento concluído com sucesso!")

                    except Exception as e:
                        st.error(f"❌ Erro ao processar: {str(e)}")

        elif not api_key:
            st.warning("⚠️ Por favor, insira sua chave da API na barra lateral.")
        elif uploaded_file is None:
            st.info("📤 Faça upload de uma imagem para começar.")

    # Exibir resultados
    if 'processed' in st.session_state and st.session_state.get('processed'):
        st.markdown("---")

        result = st.session_state['result']

        # Tabs para diferentes visualizações
        tab1, tab2, tab3 = st.tabs(["📊 Dados Estruturados", "📝 Resumo", "💾 JSON Completo"])

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
                        st.markdown(f"**Telefones:** {', '.join(medico['telefones']) if isinstance(medico['telefones'], list) else medico['telefones']}")

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
