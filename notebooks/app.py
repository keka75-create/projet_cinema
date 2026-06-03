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

TMDB_API_KEY = "db1c1e421c66aba5fe3ea45a2851e3fa"

# ===== CONFIGURATION DE LA PAGE =====
st.set_page_config(
    page_title="Cinéma Creuse",
    page_icon="🎬",
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

# ===== NAVIGATION =====
if st.session_state['admin_connecte']:
    page = st.sidebar.selectbox("Navigation", ["🏠 Accueil", "🔍 Recherche & Filtres", "🔐 Admin"])
else:
    page = st.sidebar.selectbox("Navigation", ["🏠 Accueil", "🔍 Recherche & Filtres"])
    st.sidebar.markdown("---")
    if st.sidebar.button("🔐 Accès Admin"):
        st.session_state['show_login'] = True

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

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Note IMDb", f"⭐ {note}/10")
        with col_b:
            st.metric("Popularité", f"🔥 {int(popularite)}")
        with col_c:
            votes = film.get('imdb_votes', '?')
            st.metric("Votes", f"🗳️ {int(votes):,}" if pd.notna(votes) else '?')

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

    st.markdown("<h3>🎥 Films similaires</h3>", unsafe_allow_html=True)
    with st.spinner("Calcul des recommandations..."):
        df_ml, sim = load_ml_model()
        similaires = recommend(titre, df_ml, sim, n=4)

    if len(similaires) > 0:
        cols = st.columns(4)
        for idx, (_, film_sim) in enumerate(similaires.iterrows()):
            with cols[idx]:
                poster_url = get_poster_url(film_sim.get('poster_path', ''))
                titre_sim = film_sim.get('title_fr', film_sim.get('original_title_imdb', ''))
                note_sim = round(film_sim.get('imdb_rating', 0), 1)
                annee_sim = int(film_sim.get('year', 0)) if pd.notna(film_sim.get('year')) else '?'

                st.markdown(f"""
                <div class="film-card">
                    <img src="{poster_url}" alt="{titre_sim}">
                    <div class="film-info">
                        <p class="film-title">{titre_sim}</p>
                        <p class="film-meta">{annee_sim}</p>
                        <p class="film-rating">⭐ {note_sim}/10</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Voir", key=f"sim_{idx}"):
                    st.session_state['film_selectionne'] = film_sim.to_dict()
                    st.rerun()

    st.divider()
    st.markdown("<h3>🎭 Avis spectateurs</h3>", unsafe_allow_html=True)

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

# ===== AFFICHER GRILLE =====
def afficher_grille(films_df, key_prefix):
    cols = st.columns(4)
    for idx, (_, film) in enumerate(films_df.iterrows()):
        with cols[idx % 4]:
            poster_url = get_poster_url(film.get('poster_path', ''))
            titre = film.get('title_fr', film.get('original_title_imdb', 'Titre inconnu'))
            annee = int(film.get('year', 0)) if pd.notna(film.get('year')) else '?'
            note = round(film.get('imdb_rating', 0), 1) if pd.notna(film.get('imdb_rating')) else '?'
            genre = film.get('main_genre', '') if pd.notna(film.get('main_genre')) else ''

            st.markdown('<div class="card-wrapper">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="film-card">
                <img src="{poster_url}" alt="{titre}">
                <div class="film-info">
                    <p class="film-title">{titre}</p>
                    <p class="film-meta">{annee} &nbsp;|&nbsp; {genre}</p>
                    <p class="film-rating">⭐ {note}/10</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶", key=f"{key_prefix}_{idx}"):
                st.session_state['film_selectionne'] = film.to_dict()
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    injecter_js()

# ===== PAGE ADMIN =====
def afficher_admin():
    import plotly.express as px

    st.markdown("<h2>🔐 Tableau de bord Admin</h2>", unsafe_allow_html=True)

    if st.button("🚪 Déconnexion"):
        st.session_state['admin_connecte'] = False
        st.rerun()

    st.divider()

    st.markdown("<h3>📊 KPIs du catalogue</h3>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{len(df):,}</div><div class="kpi-label">Films au catalogue</div></div>', unsafe_allow_html=True)
    with col2:
        note_moy = round(df['imdb_rating'].mean(), 2)
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">⭐ {note_moy}</div><div class="kpi-label">Note moyenne</div></div>', unsafe_allow_html=True)
    with col3:
        nb_genres = df['main_genre'].nunique()
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{nb_genres}</div><div class="kpi-label">Genres différents</div></div>', unsafe_allow_html=True)
    with col4:
        annee_min = int(df['year'].min())
        annee_max = int(df['year'].max())
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{annee_min}-{annee_max}</div><div class="kpi-label">Période couverte</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>📈 Analyses interactives</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        genres_count = df['main_genre'].value_counts().reset_index()
        genres_count.columns = ['Genre', 'Nombre']
        fig = px.bar(genres_count, x='Genre', y='Nombre', title='Distribution des genres', color='Nombre', color_continuous_scale='RdPu')
        fig.update_layout(paper_bgcolor='#22223b', plot_bgcolor='#22223b', font_color='#f2e9e4', title_font_color='#c9ada7')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        notes_genre = df.groupby('main_genre')['imdb_rating'].mean().reset_index()
        notes_genre.columns = ['Genre', 'Note moyenne']
        fig2 = px.bar(notes_genre.sort_values('Note moyenne', ascending=False), x='Genre', y='Note moyenne', title='Note moyenne par genre', color='Note moyenne', color_continuous_scale='RdPu')
        fig2.update_layout(paper_bgcolor='#22223b', plot_bgcolor='#22223b', font_color='#f2e9e4', title_font_color='#c9ada7')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        df['decennie'] = (df['year'] // 10 * 10).astype(str) + 's'
        decennie_count = df['decennie'].value_counts().sort_index().reset_index()
        decennie_count.columns = ['Décennie', 'Nombre']
        fig3 = px.line(decennie_count, x='Décennie', y='Nombre', title='Films par décennie', markers=True)
        fig3.update_traces(line_color='#c9ada7', marker_color='#c9ada7')
        fig3.update_layout(paper_bgcolor='#22223b', plot_bgcolor='#22223b', font_color='#f2e9e4', title_font_color='#c9ada7')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        top_pop = df.nlargest(10, 'tmdb_popularity')[['title_fr', 'tmdb_popularity']]
        fig4 = px.bar(top_pop, x='tmdb_popularity', y='title_fr', orientation='h', title='Top 10 films les plus populaires', color='tmdb_popularity', color_continuous_scale='RdPu')
        fig4.update_layout(paper_bgcolor='#22223b', plot_bgcolor='#22223b', font_color='#f2e9e4', title_font_color='#c9ada7', yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.markdown("<h3>💬 Analyse des avis par film</h3>", unsafe_allow_html=True)
    if len(summary_df) > 0:
        df_admin = summary_df.copy()
        df_admin['% positifs'] = (df_admin['nb_positives'] / df_admin['nb_reviews'] * 100).round(1)
        df_admin['% négatifs'] = (df_admin['nb_negatives'] / df_admin['nb_reviews'] * 100).round(1)
        df_admin = df_admin.sort_values('% positifs', ascending=False)

        fig_avis = px.bar(
            df_admin, x='title', y=['% positifs', '% négatifs'],
            title='Ratio avis positifs / négatifs par film',
            barmode='group',
            color_discrete_map={'% positifs': '#a8d5a2', '% négatifs': '#d5a2a2'}
        )
        fig_avis.update_layout(
            paper_bgcolor='#22223b', plot_bgcolor='#22223b',
            font_color='#f2e9e4', title_font_color='#c9ada7',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_avis, use_container_width=True)

        st.markdown("<h4>📋 Détail par film</h4>", unsafe_allow_html=True)
        st.dataframe(
            df_admin[['title', 'nb_reviews', '% positifs', '% négatifs', 'score_moyen']].rename(columns={'title': 'Film'}),
            use_container_width=True, hide_index=True
        )

    # KPI - EVOLUTION DURÉE DES FILMS PAR DÉCENNIE
    st.markdown("<h3>📊 Évolution durée par décennie</h3>", unsafe_allow_html=True)
    movies_tmp = df.copy()
    movies_tmp["genres_split"] = movies_tmp["genres_imdb"].str.split(",")
    movies_tmp = movies_tmp.explode("genres_split")
    movies_tmp["genres_split"] = movies_tmp["genres_split"].str.strip()

    tous_genres = sorted(movies_tmp["genres_split"].dropna().unique().tolist())
    top_6 = movies_tmp["genres_split"].value_counts().head(6).index.tolist()

    genres_choisis = st.multiselect(
        "Choisissez les genres à afficher :",
        options=tous_genres,
        default=top_6
    )

    df_kpi = (movies_tmp[movies_tmp["genres_split"].isin(genres_choisis)]
        .groupby(["decade", "genres_split"])["runtime_imdb"]
        .mean()
        .reset_index())

    fig_duree = px.line(
        df_kpi.sort_values("decade"),
        x="decade",
        y="runtime_imdb",
        color="genres_split",
        facet_col="genres_split",
        facet_col_wrap=3,
        markers=True,
        title="Durée moyenne par décennie pour les principaux genres",
        labels={"decade": "Décennie", "runtime_imdb": "Durée (min)"}
    )
    fig_duree.update_xaxes(showgrid=True)
    fig_duree.update_yaxes(showgrid=True)
    fig_duree.update_layout(
        height=700,
        title_font_size=18,
        showlegend=False,
        paper_bgcolor='#22223b',
        plot_bgcolor='#22223b',
        font_color='#f2e9e4',
        title_font_color='#c9ada7'
    )
    fig_duree.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig_duree, use_container_width=True)

    st.divider()
    st.markdown("<h3>🖼️ Graphiques de l'analyse IMDb</h3>", unsafe_allow_html=True)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    graph_folder = os.path.join(BASE_DIR, '..', 'output', 'graphiques')

    graphiques = [
        ('01_top_acteurs.png', 'Top 10 Acteurs'),
        ('02_top_realisateurs.png', 'Top 10 Réalisateurs'),
        ('03_top_compositeurs.png', 'Top 10 Compositeurs'),
        ('04_top_scenaristes.png', 'Top 10 Scénaristes'),
        ('05_top_regions.png', 'Top 10 Régions'),
        ('06_top_genres.png', 'Top 10 Genres'),
        ('07_timeline_annees.png', 'Timeline Années'),
        ('08_distribution_duree.png', 'Distribution Durée'),
        ('09_distribution_notes.png', 'Distribution Notes'),
        ('10_distribution_votes.png', 'Distribution Votes'),
        ('11_top_production_companies.png', 'Top Sociétés Production'),
        ('12_top_production_countries.png', 'Top Pays Production'),
    ]

    for i in range(0, len(graphiques), 2):
        col1, col2 = st.columns(2)
        for j, col in enumerate([col1, col2]):
            if i + j < len(graphiques):
                filename, label = graphiques[i + j]
                path = os.path.join(graph_folder, filename)
                if os.path.exists(path):
                    with col:
                        st.markdown(f"<p style='color:#c9ada7; text-align:center;'>{label}</p>", unsafe_allow_html=True)
                        st.image(path, use_container_width=True)

    st.divider()
    st.markdown("<h3>📋 Étude de marché Creuse</h3>", unsafe_allow_html=True)

    data_creuse = {
        'Année': [2019, 2020, 2021, 2022, 2023, 2024, 2025],
        'Creuse (moyenne)': [385, 200, 330, 350, 365, 375, 380],
        'France': [744, 424, 650, 673, 715, 744, 725]
    }
    fig5 = px.line(pd.DataFrame(data_creuse), x='Année', y=['Creuse (moyenne)', 'France'],
                   title='Films projetés : Creuse vs France', markers=True,
                   color_discrete_map={'Creuse (moyenne)': '#c9ada7', 'France': '#9a8c98'})
    fig5.update_layout(paper_bgcolor='#22223b', plot_bgcolor='#22223b', font_color='#f2e9e4', title_font_color='#c9ada7')
    st.plotly_chart(fig5, use_container_width=True)

    data_genres = {
        'Genre': ['Documentaire', 'Drame', 'Action', 'Comedie', 'Animation', 'Thriller', 'Comedie_dramatique', 'Science-fiction', 'Horreur'],
        'Total': [475, 439, 364, 346, 274, 211, 211, 188, 161]
    }
    fig6 = px.bar(pd.DataFrame(data_genres), x='Total', y='Genre', orientation='h',
                  title='Genres les plus représentés en France (2019-2023)',
                  color='Total', color_continuous_scale='RdPu')
    fig6.update_layout(paper_bgcolor='#22223b', plot_bgcolor='#22223b', font_color='#f2e9e4', title_font_color='#c9ada7', yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig6, use_container_width=True)

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
        st.markdown("<h2>Les films du moment</h2>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⭐ Meilleures notes", use_container_width=True):
                st.session_state['tri'] = 'notes'
                st.rerun()
        with col2:
            if st.button("🔥 Plus populaires", use_container_width=True):
                st.session_state['tri'] = 'populaires'
                st.rerun()
        with col3:
            if st.button("🆕 Plus récents", use_container_width=True):
                st.session_state['tri'] = 'recents'
                st.rerun()

        st.markdown('<div class="barre-deco"></div>', unsafe_allow_html=True)

        if st.session_state['tri'] == 'notes':
            top_films = df.sort_values('imdb_rating', ascending=False).head(20)
        elif st.session_state['tri'] == 'populaires':
            top_films = df.sort_values('tmdb_popularity', ascending=False).head(20)
        else:
            top_films = df.sort_values('year', ascending=False).head(20)

        afficher_grille(top_films, "accueil")

# ===== PAGE RECHERCHE =====
elif page == "🔍 Recherche & Filtres":

    if st.session_state['film_selectionne'] is not None:
        afficher_detail(st.session_state['film_selectionne'])
    else:
        st.markdown("<h2>Recherche & Filtres</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            recherche = st.text_input("🔍 Rechercher un film", placeholder="Ex: Inception...")
        with col2:
            genres_uniques = ['Tous'] + sorted(df['main_genre'].dropna().unique().tolist())
            genre_choisi = st.selectbox("🎭 Filtrer par genre", genres_uniques)

        st.markdown('<div class="barre-deco"></div>', unsafe_allow_html=True)

        df_filtre = df.copy()
        if recherche:
            df_filtre = df_filtre[
                df_filtre['title_fr'].str.contains(recherche, case=False, na=False) |
                df_filtre['original_title_imdb'].str.contains(recherche, case=False, na=False)
            ]
        if genre_choisi != 'Tous':
            df_filtre = df_filtre[df_filtre['main_genre'] == genre_choisi]

        st.markdown(f"<p style='color:#9a8c98;'>{len(df_filtre)} films trouvés</p>", unsafe_allow_html=True)

        afficher_grille(df_filtre.head(20), "recherche")

# ===== PAGE ADMIN =====
elif page == "🔐 Admin":
    if st.session_state['admin_connecte']:
        afficher_admin()
    else:
        st.error("Accès non autorisé !")
        st.rerun()
