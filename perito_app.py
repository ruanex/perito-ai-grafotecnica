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

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="PERITO.AI | Diamond v11", page_icon="💎", layout="wide", initial_sidebar_state="expanded")


# --- CSS DARK FORENSE ---
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        .stApp { background-color: #0e1117; color: #e0e0e0; font-family: 'Roboto', sans-serif; }
        [data-testid="stSidebar"] { background-color: #161920; border-right: 1px solid #333; }
        h1, h2, h3 { color: #00e5ff !important; font-weight: 300; letter-spacing: 1px; }
        .stButton>button { border: 1px solid #00e5ff; color: #00e5ff; background: transparent; border-radius: 4px; }
        .stButton>button:hover { background: #00e5ff; color: black; box-shadow: 0 0 10px #00e5ff; }
        div[data-baseweb="slider"] { padding-top: 20px; }
        /* Tabela de Metadados */
        .dataframe { font-family: 'Courier New', monospace; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)


local_css()

# --- ESTADO E PERSISTÊNCIA ---
if "laudo_itens" not in st.session_state: st.session_state["laudo_itens"] = []
if "pontos" not in st.session_state: st.session_state["pontos"] = []
if "logo_path" not in st.session_state: st.session_state["logo_path"] = None


def download_estado():
    return pickle.dumps({"itens": st.session_state["laudo_itens"]})


def carregar_estado(f):
    try:
        st.session_state["laudo_itens"] = pickle.load(f).get("itens", [])
        st.toast("Caso carregado.", icon="📂")
    except:
        st.error("Erro ao carregar.")


def adicionar_ao_laudo(img, tit, desc):
    # Se a imagem for um gráfico do matplotlib (figura), converte para array
    if isinstance(img, plt.Figure):
        import io
        buf = io.BytesIO()
        img.savefig(buf, format='png', bbox_inches='tight', facecolor='#0e1117')
        buf.seek(0)
        img = np.array(Image.open(buf).convert('RGB'))

    st.session_state["laudo_itens"].append({"imagem": img, "titulo": tit, "descricao": desc})
    st.toast(f"Adicionado: {tit}", icon="✅")


# --- FUNÇÕES DE IMAGEM ---
def carregar_imagem_segura(uploaded, key_suffix=""):
    """Carrega imagem e aplica rotação se necessário"""
    if uploaded is None: return None
    img = Image.open(uploaded).convert('RGB')

    with st.expander(f"🛠️ Ajustes ({key_suffix})"):
        col_rot, col_grid = st.columns(2)
        rot = col_rot.slider(f"Rotação (°)", -20.0, 20.0, 0.0, 0.1, key=f"rot_{key_suffix}")
        grid = col_grid.checkbox("Ativar Grid Milimétrico", key=f"grid_{key_suffix}")

        if rot != 0:
            img = img.rotate(-rot, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

        arr = np.array(img)

        if grid:
            # Desenha Grid
            h, w, _ = arr.shape
            step = 50  # Tamanho do quadrado
            overlay = arr.copy()
            # Linhas verticais
            for x in range(0, w, step):
                cv2.line(overlay, (x, 0), (x, h), (200, 200, 200), 1)
            # Linhas horizontais
            for y in range(0, h, step):
                cv2.line(overlay, (0, y), (w, y), (200, 200, 200), 1)
            # Blend suave
            arr = cv2.addWeighted(overlay, 0.3, arr, 0.7, 0)

    return arr


def extrair_metadados(uploaded_file):
    """Extrai EXIF da imagem original"""
    img = Image.open(uploaded_file)
    exif_data = {}
    if hasattr(img, '_getexif') and img._getexif():
        for tag, value in img._getexif().items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            exif_data[tag_name] = str(value)
    return exif_data


def gerar_histograma_comparativo(img_q, img_p):
    """Gera gráfico comparativo de distribuição de cores"""
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    colors = ('b', 'g', 'r')
    labels = ('Blue', 'Green', 'Red')

    for i, col in enumerate(colors):
        # Questionada (Linha Cheia)
        hist_q = cv2.calcHist([img_q], [i], None, [256], [0, 256])
        ax.plot(hist_q, color=col, linestyle='-', alpha=0.8, label=f'Q-{labels[i]}')

        # Padrão (Linha Pontilhada)
        hist_p = cv2.calcHist([img_p], [i], None, [256], [0, 256])
        ax.plot(hist_p, color=col, linestyle=':', alpha=0.6, label=f'P-{labels[i]}')

    ax.set_title("Espectrografia de Tinta (RGB)", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    # ax.legend() # Legenda pode poluir, opcional
    plt.tight_layout()
    return fig


def recortar_componentes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted([cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 50], key=lambda x: x[0])
    return [img[y:y + h, x:x + w] for x, y, w, h in boxes]


def criar_montagem(imgs):
    if not imgs: return None
    mh = max(i.shape[0] for i in imgs);
    tw = sum(i.shape[1] for i in imgs) + len(imgs) * 10
    mont = np.ones((mh, tw, 3), dtype=np.uint8) * 255
    cx = 0
    for i in imgs: h, w = i.shape[:2]; mont[(mh - h) // 2:(mh - h) // 2 + h, cx:cx + w] = i; cx += w + 10
    return mont


# --- PDF CUSTOMIZADO ---
def gerar_pdf(proc, nome, itens, logo_bytes=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    def header_custom():
        pdf.add_page()
        if logo_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_logo:
                tmp_logo.write(logo_bytes)
                pdf.image(tmp_logo.name, x=10, y=8, w=30)
                pdf.ln(5)
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 15, "Relatório Pericial Grafotécnico", ln=True, align='C')
        pdf.ln(10)

    header_custom()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"Processo: {proc}", ln=True)
    pdf.cell(0, 8, f"Interessado: {nome}", ln=True)
    pdf.cell(0, 8, f"Data: {np.datetime64('today', 'D')}", ln=True)
    pdf.ln(10)

    for i, item in enumerate(itens):
        if pdf.get_y() > 250: pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, f"Exame {i + 1}: {item['titulo']}", ln=True, fill=True)
        pdf.ln(2)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 6, item['descricao'])
        pdf.ln(5)

        # Tratamento de Imagem
        img_data = item['imagem']
        if isinstance(img_data, np.ndarray):
            img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                cv2.imwrite(tmp.name, img)
                h, w = img.shape[:2];
                r = w / h
                width = 170 if r < 1.5 else 190
                x_pos = (210 - width) / 2
                pdf.image(tmp.name, x=x_pos, w=width)
        pdf.ln(10)
    return bytes(pdf.output())


# --- BARRA LATERAL ---
with st.sidebar:
    c1, c2 = st.columns([1, 4])
    with c1:
        st.markdown("## 💎")
    with c2:
        st.markdown("### PERITO.AI\nDiamond v11")
    st.markdown("---")

    # 1. Arquivo
    f_load = st.file_uploader("Carregar .perito", type="perito", label_visibility="collapsed")
    if f_load: carregar_estado(f_load)
    if st.session_state["laudo_itens"]: st.download_button("💾 Salvar Caso", download_estado(), "backup.perito")

    # 2. Logo
    st.markdown("---")
    st.markdown("**Identidade Visual**")
    f_logo = st.file_uploader("Logo (PNG/JPG)", type=["png", "jpg"], key="logo_up", label_visibility="collapsed")

    # 3. Laudo
    st.markdown("---")
    st.markdown("**Emitir Laudo**")
    proc = st.text_input("Processo", "000/2026")
    nome = st.text_input("Nome", "Anônimo")
    st.info(f"Evidências: {len(st.session_state['laudo_itens'])}")

    if st.session_state["laudo_itens"]:
        logo_data = f_logo.getvalue() if f_logo else None
        pdf = gerar_pdf(proc, nome, st.session_state["laudo_itens"], logo_data)
        st.download_button("📄 PDF FINAL", pdf, "Laudo_Final.pdf", "primary")
        if st.button("Limpar Tudo"): st.session_state["laudo_itens"] = []; st.rerun()

# --- ABAS PRINCIPAIS ---
ab1, ab2, ab3, ab4, ab5, ab6, ab7 = st.tabs(
    ["GONIOMETRIA", "CONFRONTO", "METADADOS & TINTA", "SOBREPOSIÇÃO", "PRESSÃO", "FILTROS", "RECORTES"])

# 1. Goniometria
with ab1:
    st.markdown("#### 📐 Inclinação Axial & Grid")
    img = carregar_imagem_segura(st.file_uploader("Assinatura", key="u1"), "gonio")
    if img is not None:
        c1, c2 = st.columns([3, 1])
        with c1:
            val = streamlit_image_coordinates(Image.fromarray(img), key="coord1")
            if val: st.session_state["pontos"].append((val["x"], val["y"]))
        with c2:
            st.write(f"Pontos: {len(st.session_state['pontos'])}")
            if st.button("Desfazer"): st.session_state["pontos"] = []
            if len(st.session_state["pontos"]) >= 2:
                p1, p2 = st.session_state["pontos"][-2:]
                deg = math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1]))
                st.metric("Ângulo", f"{deg:.2f}°")
                vis = img.copy()
                cv2.line(vis, (p1[0], 0), (p1[0], vis.shape[0]), (50, 50, 50), 1)
                cv2.line(vis, p1, p2, (0, 255, 255), 2)
                if st.button("Adicionar"): adicionar_ao_laudo(vis, "Goniometria", f"Inclinação Axial: {deg:.2f} graus.")

# 2. Confronto
with ab2:
    st.markdown("#### ⚖️ Comparação Lado a Lado")
    c1, c2 = st.columns(2)
    i1 = carregar_imagem_segura(c1.file_uploader("Questionada", key="uq2"), "conf1")
    i2 = carregar_imagem_segura(c2.file_uploader("Padrão", key="up2"), "conf2")
    if i1 is not None and i2 is not None:
        if st.button("Gerar Comparação"):
            h = i1.shape[0];
            ratio = i2.shape[1] / i2.shape[0]
            i2r = cv2.resize(i2, (int(h * ratio), h))
            final = np.hstack((i1, np.ones((h, 10, 3), dtype=np.uint8) * 128, i2r))
            st.image(final, use_container_width=True)
            if st.button("Adicionar"): adicionar_ao_laudo(final, "Confronto", "Comparação direta.")

# 3. Metadados e Tinta (NOVA ABA)
with ab3:
    st.markdown("#### 🧬 Análise de Arquivo e Química Digital")
    c1, c2 = st.columns(2)

    with c1:
        st.info("📂 Inspetor de Metadados (EXIF)")
        u_meta = st.file_uploader("Imagem Original (sem edição)", key="umeta")
        if u_meta:
            meta = extrair_metadados(u_meta)
            if meta:
                st.json(meta)
                # Formata texto para o laudo
                meta_txt = "\n".join(
                    [f"{k}: {v}" for k, v in meta.items() if k in ['Model', 'Software', 'DateTime', 'Make']])
                if st.button("Adicionar Metadados"):
                    # Cria imagem dummy com texto para o laudo visual
                    dummy = np.ones((100, 600, 3), dtype=np.uint8) * 255
                    cv2.putText(dummy, "Metadados Extraidos (Ver texto)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 2)
                    adicionar_ao_laudo(dummy, "Metadados EXIF", f"Dados encontrados:\n{meta_txt}")
            else:
                st.warning("Nenhum metadado encontrado (Imagem limpa ou printscreen).")

    with c2:
        st.info("🧪 Espectrografia Comparativa (Histograma)")
        c2a, c2b = st.columns(2)
        uq_hist = c2a.file_uploader("Q", key="hq")
        up_hist = c2b.file_uploader("P", key="hp")

        if uq_hist and up_hist:
            iq_h = np.array(Image.open(uq_hist).convert('RGB'))
            ip_h = np.array(Image.open(up_hist).convert('RGB'))

            fig = gerar_histograma_comparativo(iq_h, ip_h)
            st.pyplot(fig)
            st.caption("Comparação de curvas RGB. Curvas muito distantes indicam canetas diferentes.")

            if st.button("Adicionar Histograma"):
                adicionar_ao_laudo(fig, "Espectrografia RGB", "Análise comparativa da composição cromática das tintas.")

# 4. Sobreposição
with ab4:
    st.markdown("#### 👻 Sobreposição")
    i1 = carregar_imagem_segura(st.file_uploader("Fundo", key="u3a"), "sob1")
    i2 = carregar_imagem_segura(st.file_uploader("Frente", key="u3b"), "sob2")
    if i1 is not None and i2 is not None:
        a = st.slider("Alpha", 0.0, 1.0, 0.5)
        i2r = cv2.resize(i2, (i1.shape[1], i1.shape[0]))
        res = cv2.addWeighted(i2r, a, i1, 1.0 - a, 0)
        st.image(res, use_container_width=True)
        if st.button("Adicionar"): adicionar_ao_laudo(res, "Sobreposição", f"Alpha: {a}")

# 5. Pressão
with ab5:
    st.markdown("#### 🔥 Mapa de Calor")
    i = carregar_imagem_segura(st.file_uploader("Assinatura", key="u4"), "pres")
    if i is not None:
        g = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
        hm = cv2.applyColorMap(cv2.convertScaleAbs(cv2.bitwise_not(g), alpha=1.5), cv2.COLORMAP_JET)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        st.image(hm, use_container_width=True)
        if st.button("Adicionar"): adicionar_ao_laudo(hm, "Pseudo-Pressão", "Análise de densidade.")

# 6. Filtros
with ab6:
    st.markdown("#### 🕵️ Filtros Ópticos")
    i = carregar_imagem_segura(st.file_uploader("Doc", key="u5"), "filt")
    if i is not None:
        f = st.selectbox("Filtro", ["Negativo", "Binarização", "Bordas"])
        if f == "Negativo":
            r = cv2.bitwise_not(i)
        elif f == "Binarização":
            _, r = cv2.threshold(cv2.cvtColor(i, cv2.COLOR_RGB2GRAY), 127, 255, cv2.THRESH_BINARY); r = cv2.cvtColor(r,
                                                                                                                     cv2.COLOR_GRAY2RGB)
        else:
            r = cv2.cvtColor(cv2.bitwise_not(cv2.Canny(i, 100, 200)), cv2.COLOR_GRAY2RGB)
        st.image(r, use_container_width=True)
        if st.button("Adicionar"): adicionar_ao_laudo(r, f"Filtro {f}", "Realce forense.")

# 7. Recortes
with ab7:
    st.markdown("#### ✂️ Segmentação")
    c1, c2 = st.columns(2)
    iq = carregar_imagem_segura(c1.file_uploader("Q", key="uq6"), "rec1")
    ip = carregar_imagem_segura(c2.file_uploader("P", key="up6"), "rec2")
    if iq is not None and ip is not None:
        if st.button("Segmentar"):
            rq, rp = recortar_componentes(iq), recortar_componentes(ip)
            fq, fp = criar_montagem(rq), criar_montagem(rp)
            if fq is not None and fp is not None:
                w = max(fq.shape[1], fp.shape[1])
                pad = lambda x, w: np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0)), constant_values=255)
                fin = np.vstack((pad(fq, w), np.ones((20, w, 3), dtype=np.uint8) * 200, pad(fp, w)))
                st.image(fin, use_container_width=True)
                st.session_state["tmp_rec"] = fin
        if "tmp_rec" in st.session_state and st.button("Adicionar"):
            adicionar_ao_laudo(st.session_state["tmp_rec"], "Recortes", "Análise morfogenética.")