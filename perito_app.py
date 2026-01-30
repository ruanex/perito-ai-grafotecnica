import streamlit as st
import cv2
import numpy as np
import math
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from fpdf import FPDF
import google.generativeai as genai
# Novas bibliotecas para o Banco de Dados
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="PERITO.CLOUD | Database", page_icon="💾", layout="wide", initial_sidebar_state="expanded")

# --- CSS (MANTIDO) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');
    .stApp { background-image: linear-gradient(rgba(10, 20, 30, 0.90), rgba(0, 5, 10, 0.98)), url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop"); background-size: cover; background-attachment: fixed; color: #fff; font-family: 'Inter', sans-serif; }
    .tool-card { background-color: rgba(20, 20, 20, 0.8); backdrop-filter: blur(5px); border: 1px solid #333; border-radius: 12px; padding: 20px; text-align: center; transition: 0.3s; height: 100%; display: flex; flex-direction: column; justify-content: space-between; }
    .tool-card:hover { border-color: #00c6ff; box-shadow: 0 10px 30px rgba(0, 198, 255, 0.15); transform: translateY(-5px); }
    div.stButton > button { background: linear-gradient(90deg, #0066cc 0%, #00ccff 100%); border:none; height:45px; font-weight:bold; color:white; border-radius:8px; width:100%; }
    header[data-testid="stHeader"] { background: transparent!important; } div[data-testid="stDecoration"] { display:none; }
    section[data-testid="stSidebar"] { width: 300px!important; background-color: #050505!important; border-right: 1px solid #004466; }
</style>
""", unsafe_allow_html=True)

# --- ESTADO ---
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "app_mode" not in st.session_state: st.session_state["app_mode"] = "Home"
if "laudo_itens" not in st.session_state: st.session_state["laudo_itens"] = []
if "pontos" not in st.session_state: st.session_state["pontos"] = []
if "api_key" not in st.session_state: st.session_state["api_key"] = ""


# --- CONEXÃO COM GOOGLE SHEETS (NOVO) ---
def salvar_no_sheets(processo, nome, evidencias):
    try:
        # Verifica se as credenciais existem nos Secrets
        if "gcp_service_account" not in st.secrets:
            st.error("⚠️ Configuração de Banco de Dados não encontrada (Secrets).")
            return

        # Conecta ao Google
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        client = gspread.authorize(creds)

        # Abre a planilha (Certifique-se que o nome é EXATAMENTE este)
        sheet = client.open("perito_db").sheet1

        # Prepara os dados
        data_atual = datetime.now().strftime("%d/%m/%Y %H:%M")
        resumo_ia = " | ".join([e['titulo'] for e in evidencias])  # Lista rápida do que foi feito

        # Adiciona a linha
        sheet.append_row([data_atual, processo, nome, len(evidencias), resumo_ia])
        st.toast("✅ Caso salvo no Banco de Dados com sucesso!", icon="💾")

    except Exception as e:
        st.error(f"Erro ao salvar no banco: {e}")


# --- IA GEMINI ---
def analisar_imagem_com_ia(imagem_array, contexto="geral"):
    if not st.session_state["api_key"]: return "⚠️ Configure a API Key."
    try:
        genai.configure(api_key=st.session_state["api_key"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        pil_img = Image.fromarray(imagem_array) if isinstance(imagem_array, np.ndarray) else None
        prompts = {"geral": "Atue como Perito. Descreva a imagem tecnicamente.",
                   "goniometria": "Analise a INCLINAÇÃO AXIAL desta assinatura."}
        res = model.generate_content([prompts.get(contexto, prompts["geral"]), pil_img])
        return res.text
    except Exception as e:
        return f"Erro na IA: {str(e)}"


# --- FUNÇÕES ---
def carregar_imagem_segura(u, k=""):
    if not u: return None
    img = Image.open(u).convert('RGB')
    with st.expander(f"🛠️ Ajustes ({k})"):
        r = st.slider("Rot", -20., 20., 0., 0.1, key=f"r_{k}")
        if r != 0: img = img.rotate(-r, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    return np.array(img)


def adicionar_ao_laudo(img, tit, desc):
    st.session_state["laudo_itens"].append({"imagem": img, "titulo": tit, "descricao": desc})
    st.toast("Adicionado ao Laudo!", icon="✅")


def gerar_pdf(proc, nome, itens):
    pdf = FPDF();
    pdf.set_auto_page_break(True, 15);
    pdf.add_page()
    pdf.set_font("Arial", "B", 20);
    pdf.cell(0, 15, "Parecer Técnico Pericial", ln=True, align='C');
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12);
    pdf.cell(0, 8, f"Processo: {proc}", ln=True);
    pdf.cell(0, 8, f"Interessado: {nome}", ln=True);
    pdf.ln(10)
    for i, item in enumerate(itens):
        if pdf.get_y() > 240: pdf.add_page()
        pdf.set_font("Arial", "B", 14);
        pdf.set_fill_color(240, 240, 240);
        pdf.cell(0, 10, f"#{i + 1}: {item['titulo']}", ln=True, fill=True);
        pdf.ln(2)
        pdf.set_font("Arial", size=10);
        pdf.multi_cell(0, 5, item['descricao']);
        pdf.ln(5)
        img = cv2.cvtColor(item['imagem'], cv2.COLOR_RGB2BGR);
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png");
        cv2.imwrite(tmp.name, img)
        h, w = img.shape[:2];
        r = w / h;
        wd = 160 if r < 1.5 else 180;
        x = (210 - wd) / 2;
        pdf.image(tmp.name, x=x, w=wd);
        pdf.ln(10)
    return bytes(pdf.output())


# --- APP ---
if not st.session_state["logged_in"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.markdown(
            "<div style='background:rgba(20,30,40,0.8);padding:40px;border-radius:20px;text-align:center;border:1px solid #00c6ff;'><h1>PERITO.DB</h1><p>SYSTEM ACCESS</p></div>",
            unsafe_allow_html=True)
        pwd = st.text_input("Credencial", type="password")
        if st.button("ENTRAR"):
            if pwd == "perito123":
                st.session_state["logged_in"] = True
            else:
                st.error("Senha incorreta")
else:
    with st.sidebar:
        if "api_key" in st.secrets:
            st.session_state["api_key"] = st.secrets["api_key"]; st.success("✅ IA Conectada")
        else:
            st.session_state["api_key"] = st.text_input("API Key", type="password")

        # Indicador de Banco de Dados
        if "gcp_service_account" in st.secrets:
            st.success("✅ Banco de Dados Online")
        else:
            st.warning("⚠️ Banco desconectado")

        st.markdown("---");
        st.metric("Evidências", len(st.session_state["laudo_itens"]))
        if st.button("Sair"): st.session_state["logged_in"] = False; st.rerun()

    if st.session_state["app_mode"] == "Home":
        st.markdown("## 📂 Dashboard Integrado");
        c1, c2, c3, c4 = st.columns(4)
        # Cards simplificados para brevidade (mas funcionais)
        c1.button("📐 Goniometria", on_click=lambda: st.session_state.update(app_mode="Goniometria"))
        c2.button("⚖️ Confronto", on_click=lambda: st.session_state.update(app_mode="Confronto"))
        c3.button("👻 Decalque", on_click=lambda: st.session_state.update(app_mode="Sobreposicao"))
        c4.button("📄 Laudo Final", on_click=lambda: st.session_state.update(app_mode="Laudo"))

    # Ferramenta de Exemplo (Goniometria)
    elif st.session_state["app_mode"] == "Goniometria":
        st.button("⬅️ Voltar", on_click=lambda: st.session_state.update(app_mode="Home"))
        img = carregar_imagem_segura(st.file_uploader("Upload", key="ug"), "g")
        if img is not None:
            st.image(img, use_container_width=True)
            if st.button("🤖 Analisar com IA"):
                txt = analisar_imagem_com_ia(img, "goniometria")
                adicionar_ao_laudo(img, "Goniometria", txt)

    # ... (Outras ferramentas omitidas para caber, mantenha as do código anterior) ...

    # --- ABA LAUDO COM BANCO DE DADOS ---
    elif st.session_state["app_mode"] == "Laudo":
        st.button("⬅️ Voltar", on_click=lambda: st.session_state.update(app_mode="Home"))
        st.header("📄 Gestão e Arquivamento")

        c1, c2 = st.columns(2)
        proc = c1.text_input("Processo Nº", placeholder="Ex: 0012345-88.2026.8.26.0000")
        nome = c2.text_input("Interessado", placeholder="Ex: João da Silva")

        st.write(f"Evidências anexadas: {len(st.session_state['laudo_itens'])}")

        c_pdf, c_save = st.columns(2)

        # Botão 1: Baixar PDF
        with c_pdf:
            if st.session_state["laudo_itens"]:
                pdf = gerar_pdf(proc, nome, st.session_state["laudo_itens"])
                st.download_button("📥 Baixar PDF", pdf, "Laudo.pdf", "primary", use_container_width=True)

        # Botão 2: Salvar no Google Sheets
        with c_save:
            if st.button("💾 Salvar Caso no Banco de Dados", use_container_width=True):
                if not proc or not nome:
                    st.warning("Preencha Processo e Nome antes de salvar.")
                else:
                    salvar_no_sheets(proc, nome, st.session_state["laudo_itens"])