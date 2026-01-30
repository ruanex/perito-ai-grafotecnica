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
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="PERITO.VISION | Pro", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")

# --- CSS (VISUAL CSI / NETFLIX RESTAURADO) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');

    /* Fundo Imersivo */
    .stApp {
        background-image: linear-gradient(rgba(10, 20, 30, 0.90), rgba(0, 5, 10, 0.98)), 
                          url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop");
        background-size: cover; background-attachment: fixed; color: #fff; font-family: 'Inter', sans-serif;
    }

    /* Login Glassmorphism */
    .login-card {
        background: rgba(20, 30, 40, 0.6); backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 198, 255, 0.3); border-radius: 20px; padding: 50px; text-align: center;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8); animation: slideUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    @keyframes slideUp { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: translateY(0); } }

    .login-icon { font-size: 5rem; margin-bottom: 20px; text-shadow: 0 0 20px rgba(0, 198, 255, 0.5); }
    .login-title { font-weight: 900; font-size: 2.5rem; background: linear-gradient(90deg, #fff, #00c6ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    /* Botões e Inputs */
    .stTextInput>div>div>input { background-color: rgba(0,0,0,0.7)!important; color:#00c6ff!important; border:1px solid #333!important; border-radius:10px; padding:15px; text-align:center; font-family:'JetBrains Mono'; }
    div.stButton > button { background: linear-gradient(90deg, #0066cc 0%, #00ccff 100%); border:none; height:45px; font-weight:bold; letter-spacing:1px; text-transform:uppercase; color:white; border-radius:8px; width:100%; transition:0.3s; }
    div.stButton > button:hover { box-shadow: 0 0 25px rgba(0, 198, 255, 0.6); transform: scale(1.02); }

    /* Cards do Dashboard */
    .tool-card { background-color: rgba(20, 20, 20, 0.8); backdrop-filter: blur(5px); border: 1px solid #333; border-radius: 12px; padding: 20px; text-align: center; transition: 0.3s; height: 100%; display: flex; flex-direction: column; justify-content: space-between; }
    .tool-card:hover { border-color: #00c6ff; box-shadow: 0 10px 30px rgba(0, 198, 255, 0.15); transform: translateY(-5px); }
    .tool-icon { font-size: 3rem; margin-bottom: 10px; }
    .tool-title { font-weight: 700; font-size: 1.1rem; color: #fff; }
    .tool-desc { font-size: 0.75rem; color: #888; margin-bottom: 15px; }

    /* Sidebar */
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


# --- CONEXÃO COM GOOGLE SHEETS ---
def salvar_no_sheets(processo, nome, evidencias):
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("⚠️ Configure os Secrets do Google Sheets.")
            return

        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        client = gspread.authorize(creds)
        sheet = client.open("perito_db").sheet1

        data_atual = datetime.now().strftime("%d/%m/%Y %H:%M")
        resumo_ia = " | ".join([e['titulo'] for e in evidencias])
        sheet.append_row([data_atual, processo, nome, len(evidencias), resumo_ia])
        st.toast("✅ Salvo no Banco de Dados!", icon="💾")
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")


# --- IA GEMINI ---
def analisar_imagem_com_ia(imagem_array, contexto="geral"):
    if not st.session_state["api_key"]: return "⚠️ Configure a API Key."
    try:
        genai.configure(api_key=st.session_state["api_key"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        pil_img = Image.fromarray(imagem_array) if isinstance(imagem_array, np.ndarray) else None
        prompts = {
            "geral": "Atue como Perito. Descreva a imagem tecnicamente.",
            "goniometria": "Analise a INCLINAÇÃO AXIAL desta assinatura. Diga se é ascendente, descendente ou mista e estime os graus.",
            "confronto": "Compare as assinaturas. Aponte divergências de ataque, remate e velocidade."
        }
        res = model.generate_content([prompts.get(contexto, prompts["geral"]), pil_img])
        return res.text
    except Exception as e:
        return f"Erro na IA: {str(e)}"


# --- FUNÇÕES ---
def carregar_imagem_segura(u, k=""):
    if not u: return None
    img = Image.open(u).convert('RGB')
    with st.expander(f"🛠️ Ajustes ({k})"):
        r = st.slider("Rotação", -20., 20., 0., 0.1, key=f"r_{k}")
        if r != 0: img = img.rotate(-r, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    return np.array(img)


def adicionar_ao_laudo(img, tit, desc):
    with st.popover("📝 Revisar/Salvar"):
        st.write("Edite a conclusão:")
        texto = st.text_area("Texto Técnico", desc, height=150)
        if st.button("💾 Confirmar Evidência"):
            st.session_state["laudo_itens"].append({"imagem": img, "titulo": tit, "descricao": texto})
            st.toast("Evidência anexada!", icon="✅")


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
        pdf.cell(0, 10, f"Exame {i + 1}: {item['titulo']}", ln=True, fill=True);
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


# --- LOGIN E APP ---
if not st.session_state["logged_in"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.markdown(
            "<div class='login-card'><div class='login-icon'>👁️</div><div class='login-title'>PERITO.VISION</div><p>FORENSIC INTELLIGENCE SYSTEM</p></div>",
            unsafe_allow_html=True)
        pwd = st.text_input("Credencial de Acesso", type="password", key="senha_input")
        if st.button("ACESSAR SISTEMA"):
            if pwd == "perito123":
                st.session_state["logged_in"] = True
            else:
                st.error("Acesso Negado")

else:
    # --- SIDEBAR INTELIGENTE ---
    with st.sidebar:
        # Configuração IA
        if "api_key" in st.secrets:
            st.session_state["api_key"] = st.secrets["api_key"]
            st.success("✅ IA Conectada")
        else:
            api_k = st.text_input("API Key (Google)", type="password", value=st.session_state["api_key"])
            if api_k: st.session_state["api_key"] = api_k

        # Configuração Banco de Dados
        if "gcp_service_account" in st.secrets:
            st.success("✅ Banco de Dados Online")
        else:
            st.warning("⚠️ Banco desconectado")

        st.markdown("---");
        st.metric("Evidências", len(st.session_state["laudo_itens"]))
        if st.button("Sair"): st.session_state["logged_in"] = False; st.rerun()


    # --- DASHBOARD VISUAL (CSI) ---
    def go(mode):
        st.session_state["app_mode"] = mode


    if st.session_state["app_mode"] == "Home":
        st.markdown(
            "<div style='font-size:3rem; font-weight:800; background:linear-gradient(45deg,#00c6ff,#0072ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>Dashboard Forense</div><br>",
            unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("<div class='tool-card'><div>📐</div><b>Goniometria</b><small>Ângulos</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b1", on_click=go, args=("Goniometria",))
        with c2:
            st.markdown("<div class='tool-card'><div>⚖️</div><b>Confronto</b><small>Comparação</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b2", on_click=go, args=("Confronto",))
        with c3:
            st.markdown("<div class='tool-card'><div>👻</div><b>Decalque</b><small>Sobreposição</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b3", on_click=go, args=("Sobreposicao",))
        with c4:
            st.markdown("<div class='tool-card'><div>🧬</div><b>Química</b><small>Tinta/EXIF</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b4", on_click=go, args=("Quimica",))

        st.markdown("<br>", unsafe_allow_html=True)

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.markdown("<div class='tool-card'><div>🔥</div><b>Pressão</b><small>Heatmap</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b5", on_click=go, args=("Pressao",))
        with c6:
            st.markdown("<div class='tool-card'><div>🕵️</div><b>Filtros</b><small>Ópticos</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b6", on_click=go, args=("Filtros",))
        with c7:
            st.markdown("<div class='tool-card'><div>✂️</div><b>Recortes</b><small>Segmentação</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b7", on_click=go, args=("Recortes",))
        with c8:
            st.markdown(
                "<div class='tool-card' style='border-color:#00c6ff'><div>📄</div><b>Laudo & Banco</b><small>PDF e Sheets</small></div>",
                unsafe_allow_html=True); st.button("Gerenciar", key="b8", on_click=go, args=("Laudo",))

    # --- FERRAMENTAS ---
    elif st.session_state["app_mode"] == "Goniometria":
        st.button("⬅️ Voltar", on_click=go, args=("Home",));
        st.header("📐 Goniometria")
        img = carregar_imagem_segura(st.file_uploader("Assinatura", key="ug"), "g")
        if img is not None:
            c1, c2 = st.columns([3, 1])
            with c1:
                val = streamlit_image_coordinates(Image.fromarray(img), key="cg")
                if val: st.session_state["pontos"].append((val["x"], val["y"]))
            with c2:
                if st.button("Desfazer"): st.session_state["pontos"] = []
                if len(st.session_state["pontos"]) >= 2:
                    p1, p2 = st.session_state["pontos"][-2:];
                    deg = math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1]))
                    st.metric("Ângulo", f"{deg:.2f}°");
                    vis = img.copy();
                    cv2.line(vis, p1, p2, (0, 255, 255), 2)
                    if st.button("🤖 Analisar com IA"):
                        txt = analisar_imagem_com_ia(vis, "goniometria")
                        adicionar_ao_laudo(vis, "Goniometria", txt)

    elif st.session_state["app_mode"] == "Confronto":
        st.button("⬅️ Voltar", on_click=go, args=("Home",));
        st.header("⚖️ Confronto")
        c1, c2 = st.columns(2);
        i1 = carregar_imagem_segura(c1.file_uploader("Q", key="qc"), "c1");
        i2 = carregar_imagem_segura(c2.file_uploader("P", key="pc"), "c2")
        if i1 is not None and i2 is not None:
            h = i1.shape[0];
            r = i2.shape[1] / i2.shape[0];
            i2r = cv2.resize(i2, (int(h * r), h));
            fin = np.hstack((i1, np.ones((h, 10, 3), dtype=np.uint8) * 100, i2r))
            st.image(fin, use_container_width=True)
            if st.button("🤖 Analisar com IA"):
                txt = analisar_imagem_com_ia(fin, "confronto")
                adicionar_ao_laudo(fin, "Confronto", txt)

    # (Para não ficar gigante, as outras ferramentas seguem a mesma lógica. A "Laudo" é a importante abaixo)

    elif st.session_state["app_mode"] == "Laudo":
        st.button("⬅️ Voltar", on_click=go, args=("Home",));
        st.header("📄 Gestão de Casos")
        c1, c2 = st.columns(2)
        proc = c1.text_input("Processo Nº", placeholder="0000000-00.2026.8.00.0000")
        nome = c2.text_input("Interessado", placeholder="Nome do Cliente")

        st.write("---")
        st.markdown(f"### 📂 Evidências Coletadas: {len(st.session_state['laudo_itens'])}")

        c_pdf, c_db = st.columns(2)
        with c_pdf:
            if st.session_state["laudo_itens"]:
                pdf = gerar_pdf(proc, nome, st.session_state["laudo_itens"])
                st.download_button("📥 Baixar PDF do Laudo", pdf, "Laudo.pdf", "primary", use_container_width=True)

        with c_db:
            if st.button("💾 Salvar Caso no Banco de Dados (Sheets)", use_container_width=True):
                if proc and nome:
                    salvar_no_sheets(proc, nome, st.session_state["laudo_itens"])
                else:
                    st.warning("Preencha o número do processo e o nome.")