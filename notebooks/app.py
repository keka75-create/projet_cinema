import streamlit as st
import pandas as pd
import os
import requests
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import ast
import plotly.express as px
import numpy as np
import plotly.express as px
import numpy as np

TMDB_API_KEY = "db1c1e421c66aba5fe3ea45a2851e3fa"

# ===== CONFIGURATION DE LA PAGE =====
st.set_page_config(
    page_title="Cinéma Creuse",
    layout="wide"
)

# ===== STYLE CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400;700&display=swap');

.stApp {
    background-color: #22223b;
    color: #f2e9e4;
    font-family: 'Lato', sans-serif;
}
h1 {
    font-family: 'Playfair Display', serif !important;
    color: #c9ada7 !important;
    text-align: center;
    font-size: 3em !important;
    letter-spacing: 2px;
    padding: 20px 0;
}
h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #c9ada7 !important;
}
.stTextInput input {
    background-color: #4a4e69 !important;
    color: #f2e9e4 !important;
    border: 1px solid #9a8c98 !important;
    border-radius: 25px !important;
    padding: 10px 20px !important;
}
.stSelectbox > div > div {
    background-color: #4a4e69 !important;
    color: #f2e9e4 !important;
    border: 1px solid #9a8c98 !important;
    border-radius: 10px !important;
}
.film-card {
    background-color: #4a4e69;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #9a8c98;
    margin-bottom: 5px;
    transition: transform 0.3s ease, border-color 0.3s ease;
    cursor: pointer;
}
.film-card:hover {
    transform: scale(1.05);
    border-color: #c9ada7;
    box-shadow: 0 8px 25px rgba(0,0,0,0.5);
}
.film-card img {
    width: 100%;
    height: 280px;
    object-fit: cover;
}
.film-info { padding: 10px; }
.film-title {
    font-family: 'Playfair Display', serif;
    color: #f2e9e4;
    font-size: 14px;
    font-weight: 700;
    margin: 0 0 5px 0;
}
.film-meta { color: #9a8c98; font-size: 12px; }
.film-rating { color: #c9ada7; font-size: 13px; font-weight: 700; }
hr { border-color: #4a4e69 !important; }
[data-testid="stSidebar"] {
    background-color: #22223b !important;
    border-right: 1px solid #4a4e69 !important;
}
.stButton button {
    background-color: #c9ada7 !important;
    color: #22223b !important;
    border: none !important;
    border-radius: 25px !important;
    font-weight: 700 !important;
    padding: 8px 25px !important;
}
.card-wrapper .stButton {
    display: none !important;
}
.barre-deco {
    height: 3px;
    background: linear-gradient(to right, #22223b, #4a4e69, #9a8c98, #4a4e69, #22223b);
    border-radius: 5px;
    margin: 10px 0 25px 0;
}
.backdrop-container {
    position: relative;
    width: 100%;
    height: 400px;
    overflow: hidden;
    border-radius: 15px;
    margin-bottom: 20px;
}
.backdrop-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.backdrop-overlay {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 60%;
    background: linear-gradient(transparent, #22223b);
}
.backdrop-titre {
    position: absolute;
    bottom: 20px; left: 25px;
    font-family: 'Playfair Display', serif;
    color: #f2e9e4;
    font-size: 2.5em;
    font-weight: 700;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
}
.avis-box {
    background-color: #22223b;
    border: 1px dashed #9a8c98;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    color: #9a8c98;
    font-style: italic;
}
.kpi-card {
    background-color: #4a4e69;
    border-radius: 10px;
    padding: 20px;
    border: 1px solid #9a8c98;
    text-align: center;
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 2em;
    font-weight: 700;
    color: #c9ada7;
}
.kpi-label {
    color: #9a8c98;
    font-size: 14px;
}
[data-testid="stMetric"] {
    background-color: #4a4e69;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #9a8c98;
}
[data-testid="stMetricValue"] { color: #c9ada7 !important; }
</style>
""", unsafe_allow_html=True)

# ===== CHARGER LES DONNÉES =====
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'catalogue_films.csv'))
    return df

# ===== CHARGER LES REVIEWS =====
@st.cache_data
def load_reviews():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    reviews = pd.read_csv(os.path.join(BASE_DIR, '..', 'output', 'reviews_with_sentiment.csv'))
    summary = pd.read_csv(os.path.join(BASE_DIR, '..', 'output', 'reviews_summary.csv'))
    return reviews, summary

# ===== CHARGER LE MODÈLE ML =====
@st.cache_data
def load_ml_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, '..', 'data', 'catalogue_films.csv'))
    tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
    overview_matrix = tfidf.fit_transform(df["overview"].fillna(""))
    genres = df["genres_imdb_list"].apply(ast.literal_eval)
    mlb_genres = MultiLabelBinarizer()
    genres_matrix = mlb_genres.fit_transform(genres)
    directors = df["directors_names"].apply(ast.literal_eval).apply(lambda x: x[0] if len(x) > 0 else "Unknown")
    ohe = OneHotEncoder(handle_unknown="ignore")
    directors_matrix = ohe.fit_transform(directors.to_frame())
    actors = df["actors_names"].apply(ast.literal_eval).apply(lambda x: x[0] if len(x) > 0 else "Unknown")
    ohe2 = OneHotEncoder(handle_unknown="ignore")
    actors_matrix = ohe2.fit_transform(actors.to_frame())
    composers = df["composers_names"].apply(ast.literal_eval).apply(lambda x: x[0] if len(x) > 0 else "Unknown")
    ohe3 = OneHotEncoder(handle_unknown="ignore")
    composers_matrix = ohe3.fit_transform(composers.to_frame())
    countries = df["production_companies_country"].fillna("").str.split(",")
    mlb_country = MultiLabelBinarizer()
    country_matrix = mlb_country.fit_transform(countries)
    studios = df["production_companies_name"].fillna("").str.split(",")
    mlb_studio = MultiLabelBinarizer()
    studio_matrix = mlb_studio.fit_transform(studios)
    year_matrix = StandardScaler().fit_transform(df[["year"]])
    X = hstack([overview_matrix, genres_matrix, directors_matrix, actors_matrix, composers_matrix, country_matrix, studio_matrix, year_matrix])
    sim = cosine_similarity(X)
    return df, sim

# ===== FONCTION RECOMMANDATION =====
def recommend(title, df, sim, n=4):
    try:
        idx = df[df["original_title_imdb"] == title].index[0]
        scores = list(enumerate(sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = scores[1:n+1]
        return df.iloc[[i[0] for i in scores]]
    except:
        return pd.DataFrame()

df = load_data()
reviews_df, summary_df = load_reviews()

# ===== FONCTIONS URLs =====
def get_poster_url(poster_path):
    if pd.notna(poster_path) and str(poster_path) != '':
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/500x750/4a4e69/c9ada7?text=Pas+d'affiche"

def get_backdrop_url(backdrop_path):
    if pd.notna(backdrop_path) and str(backdrop_path) != '':
        return f"https://image.tmdb.org/t/p/original{backdrop_path}"
    return None

# ===== RÉCUPÉRER BANDE ANNONCE =====
@st.cache_data
def get_trailer(tmdb_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}/videos"
        params = {"api_key": TMDB_API_KEY, "language": "fr-FR"}
        response = requests.get(url, params=params)
        data = response.json()
        for video in data.get('results', []):
            if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                return video['key']
        params["language"] = "en-US"
        response = requests.get(url, params=params)
        data = response.json()
        for video in data.get('results', []):
            if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                return video['key']
    except:
        pass
    return None

# ===== SESSION STATE =====
if 'page_recherche' not in st.session_state:
    st.session_state['page_recherche'] = 1
if 'derniere_recherche' not in st.session_state:
    st.session_state['derniere_recherche'] = ''
if 'dernier_genre' not in st.session_state:
    st.session_state['dernier_genre'] = 'Tous'
if 'film_selectionne' not in st.session_state:
    st.session_state['film_selectionne'] = None
if 'tri' not in st.session_state:
    st.session_state['tri'] = 'notes'
if 'admin_connecte' not in st.session_state:
    st.session_state['admin_connecte'] = False
if 'show_login' not in st.session_state:
    st.session_state['show_login'] = False

# ===== TITRE =====
st.markdown("<h1>🎬 Cinéma Creuse</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9a8c98; font-size:16px; margin-top:-20px;'>Découvrez notre sélection de films</p>", unsafe_allow_html=True)
st.divider()

# ===== SLIDESHOW =====
def afficher_slideshow():
    top_films = df.sort_values('tmdb_popularity', ascending=False).head(25)
    posters_html = ""
    for _, film in top_films.iterrows():
        poster = get_poster_url(film.get('poster_path', ''))
        titre_film = film.get('title_fr', '')
        posters_html += f"""
        <div style="min-width:130px; flex-shrink:0;">
            <img src="{poster}" style="width:130px; height:190px; object-fit:cover; border-radius:10px; opacity:0.7;">
            <p style="color:#f2e9e4; font-size:10px; text-align:center; margin-top:4px; width:130px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{titre_film}</p>
        </div>
        """
    
    # On duplique pour boucle infinie
    posters_html_double = posters_html + posters_html
    
    components.html(f"""
    <style>
    .carrousel-track {{
        display: flex;
        gap: 10px;
        animation: scroll-left 40s linear infinite;
        width: max-content;
    }}
    @keyframes scroll-left {{
        0% {{ transform: translateX(0); }}
        100% {{ transform: translateX(-50%); }}
    }}
    .carrousel-wrapper {{
        overflow: hidden;
        width: 100%;
        padding: 10px 0;
    }}
    </style>
    <div class="carrousel-wrapper">
        <div class="carrousel-track">
            {posters_html_double}
        </div>
    </div>
    """, height=230)

afficher_slideshow()

# ===== NAVIGATION =====
if st.session_state['admin_connecte']:
    page = st.sidebar.selectbox("Navigation", ["🏠 Accueil", "🔍 Recherche & Filtres", "🔐 Admin"])
else:
    page = st.sidebar.selectbox("Navigation", ["🏠 Accueil", "🔍 Recherche & Filtres"])
    st.sidebar.markdown("---")
    if st.sidebar.button("🔐 Accès Admin"):
        st.session_state['show_login'] = True

if page != st.session_state.get('page_actuelle'):
    st.session_state['film_selectionne'] = None
    st.session_state['page_actuelle'] = page

# ===== JS CLIC SUR CARTE =====
def injecter_js():
    components.html("""
    <script>
    setTimeout(() => {
        const cards = window.parent.document.querySelectorAll('.film-card');
        cards.forEach((card, index) => {
            card.style.cursor = 'pointer';
            card.addEventListener('click', () => {
                const wrappers = window.parent.document.querySelectorAll('.card-wrapper');
                if (wrappers[index]) {
                    const btn = wrappers[index].querySelector('button');
                    if (btn) btn.click();
                }
            });
        });
    }, 1000);
    </script>
    """, height=0)

def format_votes(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    elif v >= 1_000:
        return f"{v/1_000:.0f}k"
    return str(int(v))

# ===== PAGE DETAIL =====
def afficher_detail(film):
    if st.button("← Retour"):
        st.session_state['film_selectionne'] = None
        st.rerun()

    titre = film.get('original_title_imdb', 'Titre inconnu')
    titre_fr = film.get('title_fr', titre)
    annee = int(film.get('year', 0)) if pd.notna(film.get('year')) else '?'
    genres_complets = film.get('genres_imdb', '') if pd.notna(film.get('genres_imdb')) else ''
    duree = film.get('runtime_imdb', '?')
    note = round(film.get('imdb_rating', 0), 1) if pd.notna(film.get('imdb_rating')) else '?'
    popularite = round(film.get('tmdb_popularity', 0), 0) if pd.notna(film.get('tmdb_popularity')) else '?'
    overview = film.get('overview', 'Pas de description disponible.')
    tmdb_id = film.get('tmdb_id', None)

    backdrop_url = get_backdrop_url(film.get('backdrop_path', ''))
    if backdrop_url:
        st.markdown(f"""
        <div class="backdrop-container">
            <img src="{backdrop_url}" alt="{titre_fr}">
            <div class="backdrop-overlay"></div>
            <div class="backdrop-titre">{titre_fr}</div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(get_poster_url(film.get('poster_path', '')), use_container_width=True)

    with col2:
        st.markdown(f"<h2>{titre_fr}</h2>", unsafe_allow_html=True)
        if titre_fr != titre:
            st.markdown(f"<p style='color:#9a8c98; font-style:italic;'>Titre original : {titre}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#9a8c98; font-size:15px;'>📅 {annee} &nbsp;|&nbsp; ⏱️ {int(duree) if pd.notna(duree) else '?'} min &nbsp;|&nbsp; 🎭 {genres_complets}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#f2e9e4; font-size:15px; line-height:1.7;'>{overview}</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Note", f"⭐ {note}/10")
        with col_b:
            votes = film.get('imdb_votes', '?')
            st.metric("Votes", f"🗳️ {format_votes(int(votes))}" if pd.notna(votes) else '?')

    st.divider()

    st.markdown("<h3>🎬 Bande annonce</h3>", unsafe_allow_html=True)
    if pd.notna(tmdb_id):
        with st.spinner("Chargement..."):
            trailer_key = get_trailer(tmdb_id)
        if trailer_key:
            st.video(f"https://www.youtube.com/watch?v={trailer_key}")
        else:
            st.markdown('<div class="avis-box">🎬 Bande annonce non disponible</div>', unsafe_allow_html=True)

    st.divider()
    with st.expander("🎥 Films similaires", expanded=False):
        with st.spinner("Calcul des recommandations..."):
            df_ml, sim = load_ml_model()
            similaires = recommend(titre, df_ml, sim, n=10)

        if len(similaires) > 0:
            afficher_bandeau("", similaires, "similaires")

    st.divider()
    with st.expander("🎭 Avis spectateurs", expanded=False):
        film_reviews = reviews_df[
            (reviews_df['title'] == titre_fr) |
            (reviews_df['title'] == titre)
        ]

        if len(film_reviews) == 0:
            st.markdown('<div class="avis-box">Aucun avis disponible pour ce film.</div>', unsafe_allow_html=True)
        else:
            nb = len(film_reviews)
            st.markdown(f"<p style='color:#9a8c98;'>💬 {nb} avis spectateurs</p>", unsafe_allow_html=True)

            nb_affiche = st.session_state.get(f'nb_avis_{titre}', 3)

            for _, avis in film_reviews.head(nb_affiche).iterrows():
                sentiment = avis['sentiment']
                emoji = "👍" if sentiment == "positive" else "👎" if sentiment == "negative" else "😐"
                couleur = "#a8d5a2" if sentiment == "positive" else "#d5a2a2" if sentiment == "negative" else "#9a8c98"
                st.markdown(f"""
                <div style="background:#4a4e69; border-left: 4px solid {couleur}; border-radius:8px; padding:15px; margin-bottom:10px;">
                   <span style="color:{couleur}; font-weight:700;">{emoji} {sentiment.capitalize()}</span>
                    <p style="color:#f2e9e4; margin-top:8px; line-height:1.6;">{avis['content']}</p>
                </div>
                """, unsafe_allow_html=True)

            if nb_affiche < nb:
                if st.button("Voir plus d'avis", key=f"more_avis_{titre}"):
                    st.session_state[f'nb_avis_{titre}'] = nb_affiche + 3
                    st.rerun()

    if st.button("← Retour", key="retour_bas"):
        st.session_state['film_selectionne'] = None
        st.rerun()

# ===== AFFICHER GRILLE =====
def afficher_grille(films_df, key_prefix):
    # On utilise 4 colonnes pour une meilleure lisibilité sur écran large
    cols = st.columns(4)
    
    for idx, (_, film) in enumerate(films_df.iterrows()):
        with cols[idx % 4]:
            # On utilise le conteneur natif avec hauteur fixe pour harmoniser tous les blocs
            with st.container(border=True, height=500):
                poster_url = get_poster_url(film.get('poster_path', ''))
                titre = film.get('title_fr', film.get('original_title_imdb', 'Titre inconnu'))
                annee = int(film.get('year', 0)) if pd.notna(film.get('year')) else '?'
                note = round(film.get('imdb_rating', 0), 1) if pd.notna(film.get('imdb_rating')) else '?'
                
                # Image
                st.image(poster_url, use_container_width=True)
                
                # Titre tronqué si trop long pour éviter le débordement
                titre_court = titre if len(titre) < 25 else titre[:22] + "..."
                st.markdown(f"**{titre_court}**")
                
                # Infos
                st.write(f"Note : {note}/10")
                
                # Bouton "Voir plus" (plus parlant que la flèche)
                if st.button("Voir plus", key=f"{key_prefix}_{idx}"):
                    st.session_state['film_selectionne'] = film.to_dict()
                    st.rerun()

# ===== PAGE ADMIN =====
def afficher_admin():
    col_titre, col_archives, col_deco = st.columns([6, 2, 2])
    with col_titre:
        st.markdown("<h2>🔐 Tableau de bord Admin</h2>", unsafe_allow_html=True)
    with col_archives:
        if st.button("📁 Archives"):
            st.session_state['show_archives'] = not st.session_state.get('show_archives', False)
    with col_deco:
        if st.button("🚪 Déconnexion"):
            st.session_state['admin_connecte'] = False
            st.rerun()

    st.divider()

    if st.session_state.get('show_archives', False):
        st.markdown("<h3>📁 Archives — Bilans & Études</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        archives_dir = os.path.join(BASE_DIR, '..', 'archives')
        os.makedirs(archives_dir, exist_ok=True)
        uploaded = st.file_uploader("➕ Ajouter une archive", type="pdf", key="upload_archive")
        if uploaded is not None:
            with open(os.path.join(archives_dir, uploaded.name), 'wb') as f:
                f.write(uploaded.getbuffer())
            st.success(f"✅ '{uploaded.name}' ajouté !")
            st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        fichiers = [f for f in os.listdir(archives_dir) if f.endswith('.pdf')]
        if fichiers:
            for fichier in fichiers:
                chemin = os.path.join(archives_dir, fichier)
                col_dl, col_sup = st.columns([6, 1])
                with col_dl:
                    with open(chemin, 'rb') as f:
                        st.download_button(label=f"📄 {fichier}", data=f, file_name=fichier, mime="application/pdf", key=f"dl_{fichier}")
                with col_sup:
                    if st.button("🗑️", key=f"sup_{fichier}"):
                        os.remove(chemin)
                        st.rerun()
        else:
            st.markdown("<p style='color:#9a8c98; font-style:italic;'>Aucune archive disponible.</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.divider()

    # 1. SECTION KPIs
    st.markdown("<h3>📊 Vue d'ensemble</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    total_films = len(df)
    note_moy = round(df['imdb_rating'].mean(), 2) if 'imdb_rating' in df.columns else 0
    nb_genres = df['main_genre'].nunique() if 'main_genre' in df.columns else 0
    annee_min = int(df['year'].min()) if 'year' in df.columns else 0
    annee_max = int(df['year'].max()) if 'year' in df.columns else 0
    with col1:
        st.metric("Nombre de films", f"{total_films:,}")
    with col2:
        st.metric("Note moyenne", f"⭐ {note_moy}")
    with col3:
        st.metric("Genres", nb_genres)
    with col4:
        st.metric("Période", f"{annee_min}-{annee_max}")

    st.divider()

    def parse_list_safe(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        if isinstance(x, (list, tuple, set)):
            return [str(i).strip() for i in x if str(i).strip()]
        s = str(x).replace(" / ", ",").replace(" | ", ",").replace(";", ",")
        return [p.strip() for p in s.split(",") if p.strip()]

    df["_actors"]    = df["actors_names"].apply(parse_list_safe)
    df["_directors"] = df["directors_names"].apply(parse_list_safe)
    df["_writers"]   = df["writers_names"].apply(parse_list_safe)
    df["_composers"] = df["composers_names"].apply(parse_list_safe)

    # ---- GRAPHIQUE 1 : TOP N ----
    st.markdown("<h3>🎬 Top N — Acteurs / Réalisateurs / etc.</h3>", unsafe_allow_html=True)
    col_choice = st.selectbox("Famille :", ["Acteurs", "Réalisateurs", "Scénaristes", "Compositeurs"], key="topn_famille")
    col_map = {"Acteurs": "_actors", "Réalisateurs": "_directors", "Scénaristes": "_writers", "Compositeurs": "_composers"}
    top_n = st.slider("Top (Qté) :", min_value=5, max_value=50, value=10, step=5, key="topn_slider")
    col_sel = col_map[col_choice]
    exploded = df.explode(col_sel)
    vc = exploded[col_sel].dropna().value_counts().head(top_n).reset_index()
    vc.columns = ["Label", "Count"]
    vc["Label"] = vc["Label"].astype(str).str.strip("[]'\"")
    if not vc.empty:
        fig_topn = px.bar(vc.sort_values("Count"), x="Count", y="Label", orientation="h",
            color="Count", color_continuous_scale=px.colors.sequential.Viridis,
            title=f"{col_choice} — Top {top_n}")
        fig_topn.update_layout(margin=dict(l=220, r=40, t=70, b=60), height=550,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#f2e9e4')
        st.plotly_chart(fig_topn, use_container_width=True)

    st.divider()

    # ---- GRAPHIQUE 2 : HISTOGRAMME ----
    st.markdown("<h3>📊 Distribution — Durée / Notes</h3>", unsafe_allow_html=True)
    df["runtime_imdb"] = pd.to_numeric(df.get("runtime_imdb"), errors="coerce")
    df["imdb_rating"]  = pd.to_numeric(df.get("imdb_rating"),  errors="coerce")
    hist_choice = st.selectbox("Variable :", ["Distribution Durée", "Distribution Notes"], key="hist_var")
    hist_bins = st.slider("Nb classes :", min_value=10, max_value=100, value=30, step=5, key="hist_bins")
    hist_col = "runtime_imdb" if hist_choice == "Distribution Durée" else "imdb_rating"
    clean = df[hist_col].dropna()
    fig_hist = px.histogram(clean, x=clean, nbins=hist_bins,
        color_discrete_sequence=["#3B528B"], title=f"{hist_choice} — {hist_bins} classes")
    fig_hist.update_layout(height=500, margin=dict(l=60, r=40, t=60, b=60), bargap=0.05,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#f2e9e4')
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ---- GRAPHIQUE 3 : SATISFACTION VS POPULARITÉ ----
    st.markdown("<h3>📈 Aide à la décision : Satisfaction vs Popularité</h3>", unsafe_allow_html=True)
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        df_summary = pd.read_csv(os.path.join(BASE_DIR, '..', 'output', 'reviews_summary.csv'))
        col_f1, col_f2 = st.columns([2, 2])
        with col_f1:
            min_avis = st.slider("Nombre minimum d'avis :", 0, int(df_summary['nb_reviews'].max()), 5, key="min_avis_slider")
        with col_f2:
            filtre_type = st.radio("Afficher :", ["Positifs", "Négatifs", "Les deux"], horizontal=True, key="filtre_avis")
        df_filtre_avis = df_summary[df_summary['nb_reviews'] >= min_avis].copy()
        if filtre_type == "Positifs":
            df_plot = df_filtre_avis.sort_values("nb_positives", ascending=True).tail(20)
            fig_avis = px.bar(df_plot, x="nb_positives", y="title", orientation="h",
                title="Top films — Avis positifs", color_discrete_sequence=["#c9ada7"])
        elif filtre_type == "Négatifs":
            df_plot = df_filtre_avis.sort_values("nb_negatives", ascending=True).tail(20)
            fig_avis = px.bar(df_plot, x="nb_negatives", y="title", orientation="h",
                title="Top films — Avis négatifs", color_discrete_sequence=["#9a8c98"])
        else:
            df_plot = df_filtre_avis.sort_values("nb_positives", ascending=True).tail(20)
            fig_avis = px.bar(df_plot, x=["nb_positives", "nb_negatives"], y="title",
                orientation="h", barmode="group", title="Top films — Avis positifs vs négatifs",
                color_discrete_sequence=["#c9ada7", "#9a8c98"])
        fig_avis.update_layout(height=550, plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)', font_color='#f2e9e4',
            xaxis_title="Nombre d'avis", yaxis_title="Film", legend_title="Type d'avis")
        st.plotly_chart(fig_avis, use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ Fichier reviews_summary.csv introuvable.")

# ===== BANDEAU NETFLIX =====
def afficher_bandeau(titre, films_df, key_prefix):
    st.markdown(f"<h3>{titre}</h3>", unsafe_allow_html=True)
    
    films = films_df.head(20).reset_index(drop=True)
    
    if f"page_{key_prefix}" not in st.session_state:
        st.session_state[f"page_{key_prefix}"] = 0
    
    page = st.session_state[f"page_{key_prefix}"]
    debut = page * 5
    films_page = films.iloc[debut:debut+5]
    
    col_prev, col_titre, col_next = st.columns([1, 8, 1])
    with col_prev:
        if page > 0:
            if st.button("‹", key=f"prev_{key_prefix}"):
                st.session_state[f"page_{key_prefix}"] -= 1
                st.rerun()
    with col_next:
        if debut + 5 < len(films):
            if st.button("›", key=f"next_{key_prefix}"):
                st.session_state[f"page_{key_prefix}"] += 1
                st.rerun()
    
    cols = st.columns(5)
    for col_idx, (_, film) in enumerate(films_page.iterrows()):
        with cols[col_idx]:
            poster_url = get_poster_url(film.get('poster_path', ''))
            titre_film = film.get('title_fr', film.get('original_title_imdb', ''))
            note = round(film.get('imdb_rating', 0), 1)
            titre_court = titre_film[:15] + "..." if len(titre_film) > 15 else titre_film
            
            st.markdown('<div class="card-wrapper">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="film-card">
                <img src="{poster_url}" alt="{titre_film}">
                <div class="film-info">
                    <p class="film-title">{titre_court}</p>
                    <p class="film-rating">⭐ {note}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("▶", key=f"{key_prefix}_{page}_{col_idx}"):
                st.session_state['film_selectionne'] = film.to_dict()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# ===== PAGE ACCUEIL =====
if page == "🏠 Accueil":

    if st.session_state.get('show_login') and not st.session_state['admin_connecte']:
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Se connecter"):
                if username == "admin" and password == "admin123":
                    st.session_state['admin_connecte'] = True
                    st.session_state['show_login'] = False
                    st.rerun()
                else:
                    st.error("Identifiants incorrects !")
        with col2:
            if st.button("Annuler"):
                st.session_state['show_login'] = False
                st.rerun()

    elif st.session_state['film_selectionne'] is not None:
        afficher_detail(st.session_state['film_selectionne'])
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            recherche_accueil = st.text_input("🔍 Rechercher un film", placeholder="Ex: Inception...", key="recherche_accueil")
        with col2:
            tous_genres_accueil = ['Tous'] + sorted(df['genres_imdb'].dropna().str.split(',').explode().str.strip().value_counts().head(10).index.tolist())
            genre_accueil = st.selectbox("🎬 Filtrer par genre :", tous_genres_accueil, key="genre_accueil")

        if recherche_accueil and len(recherche_accueil) >= 2:
            df_recherche = df[
                df['title_fr'].str.lower().str.startswith(recherche_accueil.lower()) |
                df['original_title_imdb'].str.lower().str.startswith(recherche_accueil.lower())
            ]
            if len(df_recherche) > 0:
                suggestions = df_recherche['title_fr'].tolist()[:8]
                choix = st.selectbox("🎬 Suggestions :", ["-- Choisir --"] + suggestions, key="sugg_accueil")
                if choix != "-- Choisir --":
                    film_choisi = df[df['title_fr'] == choix].iloc[0]
                    st.session_state['film_selectionne'] = film_choisi.to_dict()
                    st.rerun()
            st.markdown(f"<p style='color:#9a8c98;'>{len(df_recherche)} films trouvés</p>", unsafe_allow_html=True)
            afficher_grille(df_recherche.head(20), "accueil_recherche")
        elif recherche_accueil:
            st.markdown("<p style='color:#9a8c98;'>Tape au moins 2 lettres...</p>", unsafe_allow_html=True)
        else:
            if genre_accueil == 'Tous':
                afficher_bandeau("⭐ Les mieux notés", df.sort_values('imdb_rating', ascending=False), "notes")
                afficher_bandeau("🔥 Les plus populaires", df.sort_values('tmdb_popularity', ascending=False), "pop")
                afficher_bandeau("🆕 Les plus récents", df.sort_values('year', ascending=False), "recents")
                
                top_genres = df['genres_imdb'].dropna().str.split(',').explode().str.strip().value_counts().head(6).index.tolist()
                for genre in top_genres:
                    films_genre = df[df['genres_imdb'].str.contains(genre, case=False, na=False)].sort_values('imdb_rating', ascending=False)
                    afficher_bandeau(f"🎬 {genre}", films_genre, f"genre_{genre}")
            else:
                films_genre = df[df['genres_imdb'].str.contains(genre_accueil, case=False, na=False)].sort_values('imdb_rating', ascending=False)
                afficher_bandeau(f"🎬 {genre_accueil}", films_genre, f"genre_filtre")

# ===== PAGE RECHERCHE =====
elif page == "🔍 Recherche & Filtres":

    if st.session_state.get('show_login') and not st.session_state['admin_connecte']:
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Se connecter"):
                if username == "admin" and password == "admin123":
                    st.session_state['admin_connecte'] = True
                    st.session_state['show_login'] = False
                    st.rerun()
                else:
                    st.error("Identifiants incorrects !")
        with col2:
            if st.button("Annuler"):
                st.session_state['show_login'] = False
                st.rerun()

    elif st.session_state['film_selectionne'] is not None:
        afficher_detail(st.session_state['film_selectionne'])
    else:
        st.markdown("<h2>Recherche & Filtres</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            recherche = st.text_input("🔍 Rechercher un film", placeholder="Ex: Inception...")
            if recherche and len(recherche) >= 2:
                suggestions_df = df[
                    df['title_fr'].str.lower().str.startswith(recherche.lower()) |
                    df['original_title_imdb'].str.lower().str.startswith(recherche.lower())
                ]
                if len(suggestions_df) > 0:
                    suggestions = suggestions_df['title_fr'].tolist()[:8]
                    choix = st.selectbox("🎬 Suggestions :", ["-- Choisir --"] + suggestions, key="sugg_recherche")
                    if choix != "-- Choisir --":
                        film_choisi = df[df['title_fr'] == choix].iloc[0]
                        st.session_state['film_selectionne'] = film_choisi.to_dict()
                        st.rerun()
        with col2:
            tous_genres = df['genres_imdb'].dropna().str.split(',').explode().str.strip().unique()
            genres_uniques = ['Tous'] + sorted(tous_genres.tolist())
            genre_choisi = st.selectbox("🎭 Filtrer par genre", genres_uniques)

        st.markdown('<div class="barre-deco"></div>', unsafe_allow_html=True)

        df_filtre = df.copy()
        df_filtre = df_filtre.sort_values('imdb_rating', ascending=False)
        if recherche:
            df_filtre = df_filtre[
                df_filtre['title_fr'].str.contains(recherche, case=False, na=False) |
                df_filtre['original_title_imdb'].str.contains(recherche, case=False, na=False)
            ]
        if genre_choisi != 'Tous':
            df_filtre = df_filtre[df_filtre['genres_imdb'].str.contains(genre_choisi, case=False, na=False)]

        st.markdown(f"<p style='color:#9a8c98;'>{len(df_filtre)} films trouvés</p>", unsafe_allow_html=True)

        films_par_page = 20
        total_pages = max(1, (len(df_filtre) - 1) // films_par_page + 1)

        if recherche != st.session_state['derniere_recherche'] or genre_choisi != st.session_state['dernier_genre']:
            st.session_state['page_recherche'] = 1
        
        st.session_state['derniere_recherche'] = recherche
        st.session_state['dernier_genre'] = genre_choisi
        
        page_actuelle = st.session_state['page_recherche']
        debut = (page_actuelle - 1) * films_par_page
        fin = debut + films_par_page
        
        afficher_grille(df_filtre.iloc[debut:fin], "recherche")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if page_actuelle > 1:
                if st.button("← Précédent"):
                    st.session_state['page_recherche'] -= 1
                    st.rerun()
        with col2:
            st.markdown(f"<p style='text-align:center; color:#9a8c98;'>Page {page_actuelle} / {total_pages}</p>", unsafe_allow_html=True)
        with col3:
            if page_actuelle < total_pages:
                if st.button("Suivant →"):
                    st.session_state['page_recherche'] += 1
                    st.rerun()

# ===== PAGE ADMIN =====
elif page == "🔐 Admin":
    if st.session_state['admin_connecte']:
        afficher_admin()
    else:
        st.error("Accès non autorisé !")
        st.rerun()