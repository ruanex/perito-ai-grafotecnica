import streamlit as st
import cv2
import numpy as np
import math
import tempfile
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
from streamlit_image_coordinates import streamlit_image_coordinates
from fpdf import FPDF
import google.generativeai as genai

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="PERITO.CLOUD | AI Vision", page_icon="👁️", layout="wide",
                   initial_sidebar_state="expanded")

# --- CSS (DESIGN CSI / NETFLIX) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');

    .stApp {
        background-image: linear-gradient(rgba(10, 20, 30, 0.90), rgba(0, 5, 10, 0.98)), 
                          url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop");
        background-size: cover; background-attachment: fixed; color: #fff; font-family: 'Inter', sans-serif;
    }

    /* Login Glass */
    .login-card {
        background: rgba(20, 30, 40, 0.6); backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 198, 255, 0.3); border-radius: 20px; padding: 50px; text-align: center;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8); animation: slideUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    @keyframes slideUp { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: translateY(0); } }

    .login-icon { font-size: 5rem; margin-bottom: 20px; text-shadow: 0 0 20px rgba(0, 198, 255, 0.5); }
    .login-title { font-weight: 900; font-size: 2.5rem; background: linear-gradient(90deg, #fff, #00c6ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    /* Input & Button */
    .stTextInput>div>div>input { background-color: rgba(0,0,0,0.7)!important; color:#00c6ff!important; border:1px solid #333!important; border-radius:10px; padding:15px; text-align:center; font-family:'JetBrains Mono'; }
    div.stButton > button { background: linear-gradient(90deg, #0066cc 0%, #00ccff 100%); border:none; height:45px; font-weight:bold; letter-spacing:1px; text-transform:uppercase; color:white; border-radius:8px; width:100%; transition:0.3s; }
    div.stButton > button:hover { box-shadow: 0 0 25px rgba(0, 198, 255, 0.6); transform: scale(1.02); }

    /* Header e Sidebar */
    header[data-testid="stHeader"] { background: transparent!important; } div[data-testid="stDecoration"] { display:none; }
    section[data-testid="stSidebar"] { width: 300px!important; background-color: #050505!important; border-right: 1px solid #004466; }

    /* Cards */
    .tool-card { background-color: rgba(20, 20, 20, 0.8); backdrop-filter: blur(5px); border: 1px solid #333; border-radius: 12px; padding: 20px; text-align: center; transition: 0.3s; height: 100%; display: flex; flex-direction: column; justify-content: space-between; }
    .tool-card:hover { border-color: #00c6ff; box-shadow: 0 10px 30px rgba(0, 198, 255, 0.15); transform: translateY(-5px); }
    .tool-icon { font-size: 3rem; margin-bottom: 10px; }
    .tool-title { font-weight: 700; font-size: 1.1rem; color: #fff; }
    .tool-desc { font-size: 0.75rem; color: #888; margin-bottom: 15px; }

</style>
""", unsafe_allow_html=True)

# --- ESTADO ---
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "app_mode" not in st.session_state: st.session_state["app_mode"] = "Home"
if "laudo_itens" not in st.session_state: st.session_state["laudo_itens"] = []
if "pontos" not in st.session_state: st.session_state["pontos"] = []
if "api_key" not in st.session_state: st.session_state["api_key"] = ""


# --- LÓGICA LOGIN ---
def check_login():
    if st.session_state["senha_input"] == "perito123":
        st.session_state["logged_in"] = True
    else:
        st.toast("🔒 Acesso Negado", icon="🚫")


def logout(): st.session_state["logged_in"] = False; st.session_state["app_mode"] = "Home"


def go_home(): st.session_state["app_mode"] = "Home"


def go_tool(n): st.session_state["app_mode"] = n


# --- INTEGRAÇÃO COM IA (GEMINI) ---
def analisar_imagem_com_ia(imagem_array, contexto="geral"):
    """Envia a imagem para o Google Gemini e recebe o laudo técnico"""
    if not st.session_state["api_key"]:
        return "⚠️ Configure a API Key na barra lateral para usar a IA."

    try:
        genai.configure(api_key=st.session_state["api_key"])
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Converter array numpy para PIL
        if isinstance(imagem_array, np.ndarray):
            pil_img = Image.fromarray(imagem_array)
        else:
            return "Erro no formato da imagem."

        prompts = {
            "geral": "Atue como Perito Grafotécnico. Descreva sucintamente a imagem: tipo de traço, pressão aparente e características visíveis.",
            "goniometria": "Atue como Perito. Analise esta assinatura com foco na INCLINAÇÃO AXIAL (Goniometria). O punho é vertical, dextrógiro ou sinistrógiro? A inclinação é constante ou variável?",
            "confronto": "Atue como Perito. Compare as duas assinaturas na imagem (Questionada e Padrão). Aponte semelhanças e divergências em: ataques, remates e morfologia. Conclua se há indícios de falsificação.",
            "pressao": "Atue como Perito. Analise este mapa de calor (pressão). Onde estão os pontos de maior entintamento? A pressão é natural (variável) ou artificial (constante/lenta)?",
            "recortes": "Analise esta segmentação de letras. Compare a morfologia (forma) das letras correspondentes entre a linha superior e inferior."
        }

        prompt_final = prompts.get(contexto, prompts["geral"])

        with st.spinner("🤖 A IA está analisando a evidência..."):
            response = model.generate_content([prompt_final, pil_img])
            return response.text

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


def adicionar_ao_laudo(img, tit, desc_inicial):
    with st.popover("📝 Revisar/Salvar"):
        st.write("Edite a conclusão antes de salvar:")
        texto = st.text_area("Texto do Laudo", desc_inicial, height=200)
        if st.button("💾 Confirmar Evidência"):
            if isinstance(img, plt.Figure):
                import io;
                buf = io.BytesIO();
                img.savefig(buf, format='png', bbox_inches='tight', facecolor='#0e1117');
                buf.seek(0);
                img = np.array(Image.open(buf).convert('RGB'))
            st.session_state["laudo_itens"].append({"imagem": img, "titulo": tit, "descricao": texto})
            st.toast("Adicionado ao Laudo!", icon="✅")


def gerar_pdf(proc, nome, itens):
    pdf = FPDF();
    pdf.set_auto_page_break(True, 15)
    pdf.add_page();
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


# --- APP ---
if not st.session_state["logged_in"]:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        st.markdown(
            "<div class='login-card'><div class='login-icon'>👁️</div><div class='login-title'>PERITO.VISION</div><p>AI-POWERED FORENSICS</p></div>",
            unsafe_allow_html=True)
        st.text_input("Credencial", type="password", key="senha_input", label_visibility="collapsed",
                      placeholder="••••••••")
        st.button("ACESSAR SISTEMA", on_click=check_login)

else:
    # --- SIDEBAR INTELIGENTE ---
    with st.sidebar:
        # Tenta pegar a chave dos segredos do servidor
        if "api_key" in st.secrets:
            st.session_state["api_key"] = st.secrets["api_key"]
            st.success("✅ IA Conectada (Chave Segura)")
        else:
            # Se não achar, pede para digitar
            st.markdown("### 🤖 Configuração IA")
            api_k = st.text_input("Cole sua Google API Key:", type="password", value=st.session_state["api_key"])
            if api_k: st.session_state["api_key"] = api_k

        st.markdown("---")
        st.markdown("### 👤 Painel")
        # ... (resto do código igual)
        st.metric("Evidências", len(st.session_state["laudo_itens"]))
        if st.session_state["laudo_itens"] and st.button("Limpar"): st.session_state["laudo_itens"] = []
        st.markdown("---")
        if st.button("Sair"): logout()

    # --- DASHBOARD ---
    if st.session_state["app_mode"] == "Home":
        st.markdown(
            "<div style='font-size:3rem; font-weight:800; background:linear-gradient(45deg,#00c6ff,#0072ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>Dashboard Vision AI</div><br>",
            unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("<div class='tool-card'><div>📐</div><b>Goniometria</b><small>Ângulos</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b1", on_click=go_tool, args=("Goniometria",))
        with c2:
            st.markdown("<div class='tool-card'><div>⚖️</div><b>Confronto</b><small>Comparação</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b2", on_click=go_tool, args=("Confronto",))
        with c3:
            st.markdown("<div class='tool-card'><div>👻</div><b>Decalque</b><small>Sobreposição</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b3", on_click=go_tool, args=("Sobreposicao",))
        with c4:
            st.markdown("<div class='tool-card'><div>🧬</div><b>Química</b><small>Tinta/EXIF</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b4", on_click=go_tool, args=("Quimica",))
        st.write("")
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.markdown("<div class='tool-card'><div>🔥</div><b>Pressão</b><small>Heatmap</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b5", on_click=go_tool, args=("Pressao",))
        with c6:
            st.markdown("<div class='tool-card'><div>🕵️</div><b>Filtros</b><small>Ópticos</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b6", on_click=go_tool, args=("Filtros",))
        with c7:
            st.markdown("<div class='tool-card'><div>✂️</div><b>Recortes</b><small>Segmentação</small></div>",
                        unsafe_allow_html=True); st.button("Abrir", key="b7", on_click=go_tool, args=("Recortes",))
        with c8:
            st.markdown(
                "<div class='tool-card' style='border-color:#00c6ff'><div>📄</div><b>Laudo</b><small>Relatório PDF</small></div>",
                unsafe_allow_html=True); st.button("Gerenciar", key="b8", on_click=go_tool, args=("Laudo",))

    # --- TOOLS ---
    elif st.session_state["app_mode"] == "Goniometria":
        st.button("⬅️ Voltar", on_click=go_home);
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
                    st.metric("Graus", f"{deg:.2f}");
                    vis = img.copy();
                    cv2.line(vis, p1, p2, (0, 255, 255), 2)

                    # BOTÃO IA
                    if st.button("🤖 Analisar com IA"):
                        analise = analisar_imagem_com_ia(vis, "goniometria")
                        adicionar_ao_laudo(vis, "Goniometria (Análise IA)", analise)

    elif st.session_state["app_mode"] == "Confronto":
        st.button("⬅️ Voltar", on_click=go_home);
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

            # BOTÃO IA
            if st.button("🤖 Analisar Confronto com IA"):
                analise = analisar_imagem_com_ia(fin, "confronto")
                adicionar_ao_laudo(fin, "Confronto (Parecer IA)", analise)

    elif st.session_state["app_mode"] == "Pressao":
        st.button("⬅️ Voltar", on_click=go_home);
        st.header("🔥 Pressão")
        img = carregar_imagem_segura(st.file_uploader("Img", key="pr"), "pr")
        if img is not None:
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
            hm = cv2.applyColorMap(cv2.convertScaleAbs(cv2.bitwise_not(g), alpha=1.5), cv2.COLORMAP_JET)
            hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB);
            st.image(hm, use_container_width=True)

            # BOTÃO IA
            if st.button("🤖 Analisar Pressão com IA"):
                analise = analisar_imagem_com_ia(hm, "pressao")
                adicionar_ao_laudo(hm, "Pressão (Parecer IA)", analise)

    # (Mantenha as outras ferramentas como Sobreposicao, Quimica, Filtros, Recortes e Laudo com a mesma lógica ou a antiga se não quiser IA nelas)
    # Por brevidade, vou colocar as outras ferramentas na versão "padrão" (sem IA dedicada) mas com o código funcionando
    elif st.session_state["app_mode"] == "Sobreposicao":
        st.button("⬅️ Voltar", on_click=go_home);
        st.header("👻 Decalque")
        i1, i2 = carregar_imagem_segura(st.file_uploader("Fundo", key="s1"), "s1"), carregar_imagem_segura(
            st.file_uploader("Frente", key="s2"), "s2")
        if i1 is not None and i2 is not None:
            a = st.slider("Opacidade", 0., 1., 0.5);
            i2r = cv2.resize(i2, (i1.shape[1], i1.shape[0]))
            res = cv2.addWeighted(i2r, a, i1, 1. - a, 0);
            st.image(res, use_container_width=True)
            if st.button("Salvar"): adicionar_ao_laudo(res, "Sobreposição", f"Opacidade {a}")

    elif st.session_state["app_mode"] == "Quimica":
        st.button("⬅️ Voltar", on_click=go_home);
        st.header("🧬 Química")
        uq, up = st.file_uploader("Q", key="hq"), st.file_uploader("P", key="hp")
        if uq and up:
            iq = np.array(Image.open(uq).convert('RGB'));
            ip = np.array(Image.open(up).convert('RGB'))
            fig, ax = plt.subplots(figsize=(6, 2));
            fig.patch.set_facecolor('#0e1117');
            ax.set_facecolor('#0e1117')
            for i, c in enumerate(['b', 'g', 'r']): ax.plot(cv2.calcHist([iq], [i], None, [256], [0, 256]), color=c,
                                                            alpha=0.8); ax.plot(
                cv2.calcHist([ip], [i], None, [256], [0, 256]), color=c, linestyle=':')
            st.pyplot(fig);
            if st.button("Salvar"): adicionar_ao_laudo(fig, "Química", "Análise RGB")

    elif st.session_state["app_mode"] == "Filtros":
        st.button("⬅️ Voltar", on_click=go_home);
        st.header("🕵️ Filtros")
        img = carregar_imagem_segura(st.file_uploader("Doc", key="fi"), "fi")
        if img is not None:
            op = st.selectbox("Filtro", ["Negativo", "Binarização", "Bordas"])
            if op == "Negativo":
                res = cv2.bitwise_not(img)
            elif op == "Binarização":
                _, res = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 127, 255,
                                       cv2.THRESH_BINARY); res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
            else:
                res = cv2.cvtColor(cv2.bitwise_not(cv2.Canny(img, 100, 200)), cv2.COLOR_GRAY2RGB)
            st.image(res, use_container_width=True)
            if st.button("Salvar"): adicionar_ao_laudo(res, f"Filtro {op}", "Processamento óptico.")

    elif st.session_state["app_mode"] == "Recortes":
        st.button("⬅️ Voltar", on_click=go_home);
        st.header("✂️ Recortes")
        c1, c2 = st.columns(2)
        iq = carregar_imagem_segura(c1.file_uploader("Q", key="rq"), "r1")
        ip = carregar_imagem_segura(c2.file_uploader("P", key="rp"), "r2")
        if iq is not None and ip is not None:
            if st.button("Segmentar"):
                def seg(im):
                    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY);
                    _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    cnts, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
                    b = sorted([cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 50], key=lambda x: x[0])
                    cr = [im[y:y + h, x:x + w] for x, y, w, h in b];
                    return cr


                def mont(cr):
                    if not cr: return None
                    mh = max(c.shape[0] for c in cr);
                    tw = sum(c.shape[1] for c in cr) + len(cr) * 10;
                    s = np.ones((mh, tw, 3), dtype=np.uint8) * 255;
                    x = 0
                    for c in cr: h, w = c.shape[:2]; s[(mh - h) // 2:(mh - h) // 2 + h, x:x + w] = c; x += w + 10
                    return s


                sq, sp = seg(iq), seg(ip);
                mq, mp = mont(sq), mont(sp)
                if mq is not None and mp is not None:
                    mw = max(mq.shape[1], mp.shape[1]);
                    pad = lambda i, w: np.pad(i, ((0, 0), (0, w - i.shape[1]), (0, 0)), constant_values=255)
                    fin = np.vstack((pad(mq, mw), np.ones((20, mw, 3), dtype=np.uint8) * 200, pad(mp, mw)))
                    st.image(fin, use_container_width=True);
                    st.session_state["tmp_rec"] = fin
            if "tmp_rec" in st.session_state:
                if st.button("Salvar"): adicionar_ao_laudo(st.session_state["tmp_rec"], "Recortes", "Segmentação")

    elif st.session_state["app_mode"] == "Laudo":
        st.button("⬅️ Voltar", on_click=go_home);
        st.header("📄 Laudo Final")
        proc = st.text_input("Processo");
        nome = st.text_input("Nome")
        if st.session_state["laudo_itens"]:
            st.write("---")
            for i, item in enumerate(st.session_state["laudo_itens"]): st.text(f"{i + 1}. {item['titulo']}")
            pdf = gerar_pdf(proc, nome, st.session_state["laudo_itens"])
            st.download_button("📥 PDF COMPLETO", pdf, "Laudo.pdf", "primary")