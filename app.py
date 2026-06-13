"""
app.py – Interface Streamlit
Projet Data – DU BigData 2025-2026
Université de Montpellier
Groupe : Kadidiatou Mohamed-Hamil-Maiga, Sonia Djounfoune, Farah El-Azhari-Tahmane, Doâa Bouasse

Structure :
    Page 1 – Présentation & Données brutes
    Page 2 – Data Visualisation
    Page 3 – Économétrie (résultats GLM)
    Page 4 – Machine Learning (Classification + Régression + Clustering)
    Page 5 – Simulateur / Prédiction interactive
"""

import io, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve,
                              mean_squared_error, mean_absolute_error, r2_score)

# 
st.set_page_config(
    page_title="Projet Data – Assurance Assistance",
    page_icon="car",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main-title {
    font-size: 2.1rem; font-weight: 600; color: #1a1a2e;
    border-left: 5px solid #e63946; padding-left: 16px; margin-bottom: 4px;
}
.sub-title { color: #666; font-size: 0.95rem; margin-bottom: 20px; padding-left: 21px; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 60%, #16213e);
    border-radius: 12px; padding: 20px 24px; color: white;
    text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.metric-card h2 { font-size: 2.1rem; font-weight: 600; margin: 0; color: #e63946; }
.metric-card p  { margin: 4px 0 0 0; font-size: 0.82rem; color: #adb5bd; }
.section-hdr {
    background: #f8f9fa; border-radius: 8px; padding: 9px 14px;
    font-weight: 600; font-size: 1.02rem; color: #1a1a2e;
    margin: 18px 0 10px 0; border-left: 4px solid #e63946;
}
.insight {
    background: #fff8f8; border-left: 4px solid #e63946;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    margin: 8px 0; font-size: 0.9rem; color: #333;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Chargement des données...")
def load_and_clean():
    def load_dossiers(path):
        with open(path, encoding="latin1") as f:
            lines = f.readlines()
        cleaned = []
        for line in lines:
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            line = line.replace('""', '"')
            cleaned.append(line)
        df = pd.read_csv(io.StringIO("\n".join(cleaned)), sep=";")
        df.columns.values[0] = "Numero_dossier_ID"
        return df

    df_d = load_dossiers(f"{DATA_DIR}/dossier.csv")
    df_t = pd.read_csv(f"{DATA_DIR}/temps.csv",     sep=";", encoding="latin1")
    df_r = pd.read_csv(f"{DATA_DIR}/ressources.csv", sep=";", encoding="latin1")
    df_r.columns.values[0] = "Matricule"
    df_t.columns.values[0] = "Numero.dossier"

    df = df_d.copy()
    df = df.drop_duplicates(subset="Numero_dossier_ID", keep="first")
    df["date.ouverture"]     = pd.to_datetime(df["date.ouverture"],     dayfirst=True, errors="coerce")
    df["date.de.survenance"] = pd.to_datetime(df["date.de.survenance"], dayfirst=True, errors="coerce")
    df = df[df["date.de.survenance"].dt.year.isin([2021, 2022])]
    df = df[df["date.de.survenance"] <= df["date.ouverture"]]
    df["heure.ouverture"]             = df["heure.ouverture"].replace("25:00:00", np.nan)
    df["Assistance.ou.Administratif"] = df["Assistance.ou.Administratif"].replace("???", np.nan)
    df["Type.d.energie"]              = df["Type.d.energie"].replace("inconnu", np.nan)
    df["Cause.intervention"]          = df["Cause.intervention"].fillna("Inconnue")
    df["annee"] = df["date.ouverture"].dt.year
    df["mois"]  = df["date.ouverture"].dt.month
    df_dossiers = df.copy()

    df_t["Date.debut.traitement"] = pd.to_datetime(df_t["Date.debut.traitement"], dayfirst=True, errors="coerce")
    df_t = df_t[df_t["duree.corrigee"].isna() | (df_t["duree.corrigee"] <= 86400)]
    df_temps = df_t[df_t["duree.corrigee"].notna()].copy()

    df_r["Date.presence"] = pd.to_datetime(df_r["Date.presence"], dayfirst=True, errors="coerce")
    df_r["Duree.travail"] = df_r["Duree.travail"].clip(upper=7.3334)
    df_ressources = df_r.copy()

    return df_dossiers, df_temps, df_ressources

try:
    df_dossiers, df_temps, df_ressources = load_and_clean()
    DATA_OK = True
except Exception as e:
    DATA_OK = False; DATA_ERROR = str(e)

@st.cache_data(show_spinner="Préparation des features ML...")
def build_ml_base(_dd, _dt, _dr):
    df_temps_agg = (
        _dt.groupby("Numero.dossier")["duree.corrigee"]
        .agg(duree_totale="sum", nb_agents="count", duree_moyenne="mean")
        .reset_index().rename(columns={"Numero.dossier": "Numero_dossier_ID"})
    )
    df_agent = (
        _dt.drop_duplicates(subset="Numero.dossier", keep="first")
        [["Numero.dossier", "Matricule"]]
        .rename(columns={"Numero.dossier": "Numero_dossier_ID"})
        .merge(
            _dr[["Matricule","Population","Type.de.contrat","Experience","Lieu.travail"]]
            .drop_duplicates(subset="Matricule"), on="Matricule", how="left"
        )
    )
    df_ml = (
        _dd.merge(df_temps_agg, on="Numero_dossier_ID", how="left")
        .merge(df_agent[["Numero_dossier_ID","Population","Type.de.contrat","Experience","Lieu.travail"]],
               on="Numero_dossier_ID", how="left")
    )
    cat_cols = ["Client","Formule","Cause.intervention","Type.d.energie",
                "Outil.d.assistance","Assistance.ou.Administratif",
                "Population","Type.de.contrat","Lieu.travail"]
    df_enc = df_ml.copy()
    label_encoders = {}
    for col in cat_cols:
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].fillna("Inconnu").astype(str)
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col])
            label_encoders[col] = le
    for col in ["duree_totale","nb_agents","duree_moyenne","Experience","mois","annee"]:
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].fillna(df_enc[col].median())
    return df_enc, label_encoders

#  Sidebar 
with st.sidebar:
    st.markdown("##  Projet Data")
    st.markdown("**DU BigData 2025-2026**  \nUniversité de Montpellier")
    st.markdown("---")
    page = st.radio("Navigation", [
        "Presentation", "DataViz",
        "Econometrie", "Machine Learning", "Simulateur"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Groupe :**")
    st.markdown("- Kadidiatou\n- Sonia\n- Farah\n- Doâa")

TOPS = ["TOP.D.R","TOP.VR","TOP.Rappat.valide","TOP.Poursuite","TOP.Recup","TOP.Autres.Garanties"]

if page == "Presentation":
    st.markdown('<div class="main-title">Projet Data – Assurance Assistance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">DU BigData 2025-2026 · Université de Montpellier</div>', unsafe_allow_html=True)
    st.markdown("""
    Cette application présente l'analyse complète des données d'assurance assistance (2021-2022).
    Naviguez via le menu latéral pour explorer les différentes sections.
    """)

    if not DATA_OK:
        st.error(f" Impossible de charger les données depuis `{DATA_DIR}/`.\n\nErreur : {DATA_ERROR}")
        st.info("Vérifiez que `dossier.csv`, `temps.csv` et `ressources.csv` se trouvent dans le dossier `data/` à la racine du projet.")
        st.stop()

    st.markdown('<div class="section-hdr"> Chiffres clés après nettoyage</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><h2>{len(df_dossiers):,}</h2><p>Dossiers d\'assistance</p></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><h2>{len(df_temps):,}</h2><p>Entrées de temps</p></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><h2>{len(df_ressources):,}</h2><p>Présences agents</p></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><h2>11</h2><p>Anomalies identifiées</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr"> Tableau des anomalies identifiées</div>', unsafe_allow_html=True)
    anom = pd.DataFrame([
        {"Table":"Dossiers","Anomalie":"Doublons sur Numero_dossier_ID","Retraitement":"Suppression (1ère occurrence conservée)"},
        {"Table":"Dossiers","Anomalie":"Heure ouverture = '25:00:00'","Retraitement":"Remplacement par NaN"},
        {"Table":"Dossiers","Anomalie":"Date survenance hors 2021-2022","Retraitement":"Exclusion des lignes"},
        {"Table":"Dossiers","Anomalie":"Assistance.ou.Administratif = '???'","Retraitement":"Remplacement par NaN"},
        {"Table":"Dossiers","Anomalie":"Date survenance > Date ouverture (7 cas)","Retraitement":"Exclusion des lignes"},
        {"Table":"Dossiers","Anomalie":"Type.d.energie = 'inconnu'","Retraitement":"Remplacement par NaN"},
        {"Table":"Dossiers","Anomalie":"Formule manquante","Retraitement":"Conservation (non obligatoire)"},
        {"Table":"Dossiers","Anomalie":"Cause.intervention manquante","Retraitement":"Étiquette 'Inconnue'"},
        {"Table":"Temps",   "Anomalie":"duree.corrigee NaN (dossiers Test, 11%)","Retraitement":"Isolés, hors périmètre"},
        {"Table":"Temps",   "Anomalie":"Durée > 86 400s (> 24h)","Retraitement":"Exclusion (aberrantes)"},
        {"Table":"Ressources","Anomalie":"Duree.travail > 7.33h","Retraitement":"Plafonnement à 7.33h"},
    ])
    st.dataframe(anom, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-hdr"> Aperçu des données nettoyées</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Dossiers","Temps","Ressources"])
    with t1: st.dataframe(df_dossiers.head(10), use_container_width=True)
    with t2: st.dataframe(df_temps.head(10),    use_container_width=True)
    with t3: st.dataframe(df_ressources.head(10), use_container_width=True)

# 
#  PAGE 2 – DATA VIZ
# 
elif page == "DataViz":
    st.markdown('<div class="main-title">Data Visualisation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analyses descriptives univariées et multivariées</div>', unsafe_allow_html=True)
    if not DATA_OK: st.error("Données non disponibles."); st.stop()

    TOPS_P = [t for t in TOPS if t in df_dossiers.columns]

    viz = st.selectbox("Choisir une visualisation :", [
        "Causes d'intervention",
        "Services déclenchés (TOP)",
        "Migration MCS → Higgins par année",
        "Distribution des durées de traitement",
        "CAC vs CAS – Durée de traitement",
        "Télétravail vs Site",
        "Répartition des types de contrat",
        "Matrice de corrélation Spearman (services)",
    ])

    fig, ax = plt.subplots(figsize=(10, 5))

    if viz == "Causes d'intervention":
        counts = df_dossiers["Cause.intervention"].value_counts()
        counts.plot(kind="bar", ax=ax, color="#e63946", edgecolor="white")
        ax.set_title("Causes d'intervention", fontweight="bold"); ax.set_ylabel("Nb dossiers")
        plt.xticks(rotation=35, ha="right"); plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="insight"> La panne mecanique represente environ 67% des dossiers ce qui en fait la cause largement majoritaire. Les accidents arrivent en deuxieme position avec 16% et les problemes de cles carburant ou crevaison representent 12%. Cote energie le diesel domine avec 54% des vehicules assistes contre 35% pour lessence.</div>', unsafe_allow_html=True)

    elif viz == "Services déclenchés (TOP)":
        if TOPS_P:
            taux = df_dossiers[TOPS_P].mean() * 100
            taux.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#457b9d", edgecolor="white")
            ax.set_title("Taux de présence des services (%)", fontweight="bold"); ax.set_ylabel("%")
            plt.xticks(rotation=35, ha="right"); plt.tight_layout(); st.pyplot(fig)
            st.markdown('<div class="insight"> Le depannage remorquage est declenche dans 92.4% des dossiers ce qui en fait un service quasi systematique. Le vehicule de remplacement ne concerne que 10% des dossiers mais represente un cout important pour lentreprise, cest pourquoi on la choisi comme variable cible pour la prediction en machine learning.</div>', unsafe_allow_html=True)

    elif viz == "Migration MCS → Higgins par année":
        if "Outil.d.assistance" in df_dossiers.columns and "annee" in df_dossiers.columns:
            pct = (df_dossiers.groupby(["annee","Outil.d.assistance"]).size().unstack(fill_value=0)
                   .pipe(lambda d: d.div(d.sum(axis=1), axis=0) * 100))
            pct.plot(kind="bar", ax=ax, color=["#1d3557","#e63946"], edgecolor="white")
            ax.set_title("Outils d'assistance par année (%)", fontweight="bold"); ax.set_ylabel("%")
            ax.tick_params(axis="x", rotation=0); plt.tight_layout(); st.pyplot(fig)
            st.markdown('<div class="insight"> La migration vers le nouvel outil Higgins est clairement visible entre les deux annees. En 2021 Higgins ne representait que 17.7% des dossiers contre 82.3% pour MCS. En 2022 sa part est montee a 28% soit une progression de 10 points en un an confirmant le deploiement progressif.</div>', unsafe_allow_html=True)

    elif viz == "Distribution des durées de traitement":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df_temps["duree.corrigee"].hist(bins=100, ax=axes[0], color="#457b9d", edgecolor="white")
        axes[0].set_title("Distribution globale"); axes[0].set_xlabel("Durée (s)")
        df_temps[df_temps["duree.corrigee"] < 2000]["duree.corrigee"].hist(bins=100, ax=axes[1], color="#2a9d8f", edgecolor="white")
        axes[1].set_title("Zoom durées < 2 000s"); axes[1].set_xlabel("Durée (s)")
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="insight"> La duree de traitement a une mediane denviron 188 secondes soit 3 minutes et une moyenne de 312 secondes soit 5 minutes. Cet ecart montre une forte asymetrie a droite, la majorite des dossiers sont traites rapidement mais quelques dossiers complexes tirent la moyenne vers le haut.</div>', unsafe_allow_html=True)

    elif viz == "CAC vs CAS – Durée de traitement":
        dm = df_temps.merge(df_ressources[["Matricule","Population"]].drop_duplicates("Matricule"), on="Matricule", how="left").dropna(subset=["Population"])
        groups = [dm[dm["Population"] == p]["duree.corrigee"].dropna() for p in ["CAC","CAS"]]
        ax.boxplot(groups, labels=["CAC","CAS"],
                   boxprops=dict(color="#1d3557"), medianprops=dict(color="#e63946", linewidth=2))
        ax.set_title("Durée de traitement : CAC vs CAS", fontweight="bold"); ax.set_ylabel("Durée (s)"); ax.set_ylim(0, 3000)
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="insight"> CAS passent ~2.2× plus de temps que les CAC (médiane 229s vs 84s) — Kruskal-Wallis p < 0.0001. Cohérent avec leurs rôles distincts.</div>', unsafe_allow_html=True)

    elif viz == "Télétravail vs Site":
        dm = df_temps.merge(df_ressources[["Matricule","Lieu.travail"]].drop_duplicates("Matricule"), on="Matricule", how="left").dropna(subset=["Lieu.travail"])
        groups = [dm[dm["Lieu.travail"] == l]["duree.corrigee"].dropna() for l in ["TELE","SITE"]]
        ax.boxplot(groups, labels=["Télétravail","Site"],
                   boxprops=dict(color="#1d3557"), medianprops=dict(color="#e63946", linewidth=2))
        ax.set_title("Durée : Télétravail vs Site", fontweight="bold"); ax.set_ylabel("Durée (s)"); ax.set_ylim(0, 3000)
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="insight"> Les agents en teletravail ont une duree de traitement legerement superieure denviron 20 secondes en mediane par rapport au travail sur site. Le test statistique confirme que cette difference est significative mais elle reste faible en amplitude.</div>', unsafe_allow_html=True)

    elif viz == "Répartition des types de contrat":
        counts = df_ressources["Type.de.contrat"].value_counts()
        counts.plot(kind="pie", ax=ax, autopct="%1.1f%%",
                    colors=["#1d3557","#457b9d","#a8dadc"],
                    wedgeprops=dict(edgecolor="white", linewidth=2))
        ax.set_title("Types de contrat – Ressources", fontweight="bold"); ax.set_ylabel("")
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="insight"> 63% CDI, 27% CDD, 10% saisonniers — organisation reflétant les pics saisonniers de l\'activité d\'assistance.</div>', unsafe_allow_html=True)

    elif viz == "Matrice de corrélation Spearman (services)":
        if TOPS_P:
            corr = df_dossiers[TOPS_P].corr(method="spearman")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
            ax.set_title("Corrélation Spearman – Services", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig)
            st.markdown('<div class="insight"> TOP.Rappat.valide et TOP.Recup sont corrélés positivement — accompagnent souvent les dossiers complexes à immobilisation prolongée.</div>', unsafe_allow_html=True)

# 
#  PAGE 3 – ÉCONOMÉTRIE
# 
elif page == "Econometrie":
    st.markdown('<div class="main-title">Économétrie – Résultats GLM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Régression OLS · GLM Logistique · ACP · Clustering</div>', unsafe_allow_html=True)
    if not DATA_OK: st.error("Données non disponibles."); st.stop()

    TOPS_P = [t for t in TOPS if t in df_dossiers.columns]

    section = st.selectbox("Choisir une analyse :", [
        "Régression OLS – Durée de traitement",
        "GLM Logistique – TOP D/R",
        "ACP – Plan factoriel (TOP services)",
        "K-Means – Segmentation (services TOP)",
    ])

    if section == "Régression OLS – Durée de traitement":
        st.markdown('<div class="section-hdr">Variable cible : log(duree.corrigee) | Méthode : OLS</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Indicateur": ["R²","R² ajusté","F-statistique","p-value (F)","RMSE test (log)"],
            "Valeur":     ["0.0524","0.0524","3 652,82","< 0.0001","1.36"]
        }), use_container_width=True, hide_index=True)
        st.markdown("**Coefficients et interprétations :**")
        coefs = pd.DataFrame({
            "Variable":   ["Constante","Experience","nb_services","is_CAS","is_TELE","is_Higgins"],
            "Coef.":      [4.21, 0.000034, -0.050, 0.790, -0.060, -0.260],
            "Sig. (5%)":  ["","","","","",""],
            "Interprétation": [
                "Valeur de base",
                "Effet faible mais significatif",
                "Dossiers multi-services souvent pré-remplis",
                "CAS 2.2× plus long que CAC (e^0.79)",
                "Télétravail réduit légèrement la durée (-6%)",
                "Higgins réduit la durée de ~23% vs MCS"
            ]
        })
        st.dataframe(coefs, use_container_width=True, hide_index=True)
        st.markdown('<div class="insight"> Le modele de regression lineaire est globalement significatif avec une F statistique de 3653. Cependant le R2 nest que de 5.2%. Le facteur le plus determinant est le role de lagent, un CAS passe 2.2 fois plus longtemps sur un dossier quun CAC. Loutil Higgins reduit la duree denviron 23% par rapport a MCS.</div>', unsafe_allow_html=True)

    elif section == "GLM Logistique – TOP D/R":
        st.markdown('<div class="section-hdr">Variable cible : TOP.D.R (0/1) | Lien logit</div>', unsafe_allow_html=True)
        odds = pd.DataFrame({
            "Variable":     ["Clés/Carburant/Crevaison","Panne mécanique","Accident","is_Administratif","nb_services_hors_DR","is_Higgins"],
            "Odds Ratio":   [9.78, 7.77, 6.03, 0.40, 0.44, 1.12],
            "Sig. (5%)":    ["","","","","",""],
            "Interprétation": [
                "Multiplie la proba de D/R x 9.8",
                "x 7.8 -- panne necessite quasi toujours intervention physique",
                "x 6.0 -- certains accidents permettent de rouler",
                "divise par 2.5 -- remboursements différés sans intervention immédiate",
                "Chaque service supp. réduit la proba de D/R",
                "Higgins legerement associe a plus de D/R (+12%)"
            ]
        })
        st.dataframe(odds, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#e63946" if v > 1 else "#457b9d" for v in odds["Odds Ratio"]]
        ax.barh(odds["Variable"], odds["Odds Ratio"], color=colors, edgecolor="white")
        ax.axvline(1, color="black", linestyle="--", lw=1.5, label="OR=1 (pas d'effet)")
        ax.set_title("Odds Ratios – GLM Logistique (TOP D/R)", fontweight="bold")
        ax.set_xlabel("Odds Ratio"); ax.legend(); plt.tight_layout(); st.pyplot(fig)

    elif section == "ACP – Plan factoriel (TOP services)":
        if len(TOPS_P) < 2:
            st.warning("Colonnes TOP insuffisantes.")
        else:
            X_acp = df_dossiers[TOPS_P].dropna()
            X_sc  = StandardScaler().fit_transform(X_acp)
            pca   = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_sc)
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.15, s=5, color="#457b9d")
            ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
            ax.set_xlabel(f"CP1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"CP2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax.set_title("ACP – Plan factoriel TOP services", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig)
            st.metric("Variance expliquée (2 axes)", f"{pca.explained_variance_ratio_.sum()*100:.1f}%")
            st.markdown('<div class="insight"> Le plan factoriel oppose les dossiers avec services de mobilité (VR, rapatriement) aux dossiers de dépannage simple.</div>', unsafe_allow_html=True)

    elif section == "K-Means – Segmentation (services TOP)":
        if len(TOPS_P) < 2:
            st.warning("Colonnes TOP insuffisantes.")
        else:
            X_acp = df_dossiers[TOPS_P].dropna()
            X_sc  = StandardScaler().fit_transform(X_acp)
            k     = st.slider("Nombre de clusters K", 2, 6, 3)
            km    = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_sc)
            profil = pd.DataFrame(X_acp.values, columns=TOPS_P)
            profil["Cluster"] = labels
            profil_moy = profil.groupby("Cluster").mean() * 100
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(profil_moy, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, ax=ax)
            ax.set_title(f"Profil moyen des {k} clusters (%)", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig)
            dist = pd.Series(labels).value_counts().sort_index().reset_index()
            dist.columns = ["Cluster","Nb dossiers"]
            st.dataframe(dist, use_container_width=True, hide_index=True)
            st.markdown('<div class="insight"> 3 profils stables : dépannage simple (83%), dossiers complexes avec rapatriement (10%), dossiers administratifs sans D/R (7%). CAH confirme ce découpage.</div>', unsafe_allow_html=True)

# 
#  PAGE 4 – MACHINE LEARNING
# 
elif page == "Machine Learning":
    st.markdown('<div class="main-title">Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Classification · Régression · Clustering K-Means</div>', unsafe_allow_html=True)
    if not DATA_OK: st.error("Données non disponibles."); st.stop()

    df_enc, label_encoders = build_ml_base(df_dossiers, df_temps, df_ressources)

    ml_section = st.selectbox("Choisir un modèle :", [
        "Classification – Prédire TOP VR",
        "Régression – Estimer la durée de traitement",
        "Clustering K-Means – Segmentation dossiers",
    ])

    FEAT_CLF = ["Client","Formule","Cause.intervention","Type.d.energie",
                "Outil.d.assistance","Assistance.ou.Administratif",
                "mois","annee","duree_totale","nb_agents","duree_moyenne",
                "Experience","Population","Type.de.contrat","Lieu.travail",
                "TOP.D.R","TOP.Rappat.valide","TOP.Poursuite","TOP.Autres.Garanties"]
    FEAT_REG  = ["Client","Formule","Cause.intervention","Type.d.energie",
                 "Outil.d.assistance","Assistance.ou.Administratif",
                 "mois","annee","nb_agents",
                 "Experience","Population","Type.de.contrat","Lieu.travail",
                 "TOP.D.R","TOP.VR","TOP.Rappat.valide","TOP.Poursuite","TOP.Autres.Garanties"]

    #  Classification 
    if ml_section == "Classification – Prédire TOP VR":
        st.markdown('<div class="section-hdr">Objectif : anticiper le déclenchement d\'un véhicule de remplacement</div>', unsafe_allow_html=True)
        feat_ok = [f for f in FEAT_CLF if f in df_enc.columns]
        if "TOP.VR" not in df_enc.columns: st.error("Colonne TOP.VR introuvable."); st.stop()
        X, y = df_enc[feat_ok], df_enc["TOP.VR"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        sc = StandardScaler(); X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
        with st.spinner("Entraînement des 5 modèles..."):
            clfs = {
                "Régression Logistique": (LogisticRegression(max_iter=500, class_weight="balanced", random_state=42), True),
                "Arbre de Décision":     (DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42), False),
                "Random Forest":         (RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42), False),
                "Gradient Boosting":     (GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42), False),
                "KNN":                   (KNeighborsClassifier(n_neighbors=9, n_jobs=-1), True),
            }
            res = {}
            for name, (m, scaled) in clfs.items():
                Xtr_ = X_tr_sc if scaled else X_tr; Xte_ = X_te_sc if scaled else X_te
                m.fit(Xtr_, y_tr)
                yp = m.predict(Xte_); yprob = m.predict_proba(Xte_)[:,1]
                rpt = classification_report(y_te, yp, output_dict=True, zero_division=0)
                res[name] = {"model":m,"scaled":scaled,
                    "accuracy": rpt["accuracy"],
                    "precision":rpt.get("1",{}).get("precision",0),
                    "recall":   rpt.get("1",{}).get("recall",0),
                    "f1":       rpt.get("1",{}).get("f1-score",0),
                    "auc":      roc_auc_score(y_te, yprob),
                    "y_pred":yp,"y_proba":yprob}

        df_r = pd.DataFrame([
            {"Modèle":n,"Accuracy":v["accuracy"],"Precision":v["precision"],
             "Recall":v["recall"],"F1":v["f1"],"AUC-ROC":v["auc"]}
            for n,v in res.items()]).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
        st.dataframe(df_r.style.format("{:.4f}", subset=["Accuracy","Precision","Recall","F1","AUC-ROC"]),
                     use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        for name, v in res.items():
            fpr, tpr, _ = roc_curve(y_te, v["y_proba"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={v['auc']:.3f})", linewidth=1.5)
        ax.plot([0,1],[0,1],"k--",lw=1); ax.set_title("Courbes ROC – TOP VR", fontweight="bold")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig)

        rf_m = res["Random Forest"]["model"]
        fi = pd.Series(rf_m.feature_importances_, index=feat_ok).sort_values(ascending=False).head(12)
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        fi.sort_values().plot(kind="barh", ax=ax2, color="#e63946", edgecolor="white")
        ax2.set_title("Top 12 variables – Random Forest (TOP VR)", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig2)
        st.markdown('<div class="insight"> Random Forest et Gradient Boosting obtiennent les meilleures AUC. Durée totale et nb_agents sont les features les plus prédictives — les dossiers complexes nécessitent plus souvent un VR.</div>', unsafe_allow_html=True)

    #  Régression 
    elif ml_section == "Régression – Estimer la durée de traitement":
        st.markdown('<div class="section-hdr">Objectif : estimer la durée totale de traitement d\'un dossier</div>', unsafe_allow_html=True)
        feat_ok = [f for f in FEAT_REG if f in df_enc.columns]
        if "duree_totale" not in df_enc.columns: st.error("Colonne duree_totale introuvable."); st.stop()
        df_rg = df_enc[feat_ok + ["duree_totale"]].dropna(subset=["duree_totale"])
        X = df_rg[feat_ok]; y_log = np.log1p(df_rg["duree_totale"])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y_log, test_size=0.25, random_state=42)
        sc = StandardScaler(); X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
        with st.spinner("Entraînement des modèles de régression..."):
            regs = {
                "Régression Linéaire":     (LinearRegression(), True),
                "Ridge (α=1)":             (Ridge(alpha=1.0), True),
                "Random Forest Reg.":      (RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42), False),
                "Gradient Boosting Reg.":  (GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42), False),
            }
            res_r = {}
            for name, (m, scaled) in regs.items():
                Xtr_ = X_tr_sc if scaled else X_tr; Xte_ = X_te_sc if scaled else X_te
                m.fit(Xtr_, y_tr)
                yp_log = m.predict(Xte_)
                res_r[name] = {
                    "r2":     r2_score(y_te, yp_log),
                    "rmse":   np.sqrt(mean_squared_error(np.expm1(y_te), np.expm1(yp_log))),
                    "mae":    mean_absolute_error(np.expm1(y_te), np.expm1(yp_log)),
                    "model":  m, "yp_log": yp_log
                }
        df_rr = pd.DataFrame([{"Modèle":n,"R²":v["r2"],"RMSE (s)":v["rmse"],"MAE (s)":v["mae"]} for n,v in res_r.items()]).sort_values("R²", ascending=False).reset_index(drop=True)
        st.dataframe(df_rr.style.format({"R²":"{:.4f}","RMSE (s)":"{:.0f}","MAE (s)":"{:.0f}"}), use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors_ = ["#e63946","#457b9d","#2a9d8f","#264653"]
        ax.bar(df_rr["Modèle"], df_rr["R²"], color=colors_[:len(df_rr)], edgecolor="white")
        for i,v in enumerate(df_rr["R²"]): ax.text(i, v+0.002, f"{v:.3f}", ha="center", fontsize=10)
        ax.set_title("R² par modèle – Régression durée", fontweight="bold"); ax.set_ylim(0, 1)
        plt.xticks(rotation=15); plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="insight"> Gradient Boosting offre le meilleur R². Les R² modérés reflètent la part de variabilité non capturée (localisation, météo, prestataires). Transformation log confirmée nécessaire.</div>', unsafe_allow_html=True)

    #  Clustering 
    elif ml_section == "Clustering K-Means – Segmentation dossiers":
        st.markdown('<div class="section-hdr">Objectif : identifier des profils-types de dossiers</div>', unsafe_allow_html=True)
        FEAT_CLU = ["Cause.intervention","Type.d.energie","Outil.d.assistance",
                    "mois","annee","duree_totale","nb_agents","duree_moyenne",
                    "Experience","Population","Type.de.contrat"]
        feat_ok = [f for f in FEAT_CLU if f in df_enc.columns]
        df_clu = df_enc[feat_ok].dropna()
        X_sc = StandardScaler().fit_transform(df_clu)
        k = st.slider("Nombre de clusters K", 2, 8, 4)
        with st.spinner("Clustering..."):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_sc)
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_sc)

        idx = np.random.choice(len(X_pca), min(3000, len(X_pca)), replace=False)
        pal = ["#e63946","#457b9d","#2a9d8f","#264653","#e9c46a","#f4a261","#a8dadc","#f1faee"]
        fig, ax = plt.subplots(figsize=(10, 6))
        for cl in range(k):
            mask = labels[idx] == cl
            ax.scatter(X_pca[idx][mask,0], X_pca[idx][mask,1], label=f"Cluster {cl}", alpha=0.5, s=15, color=pal[cl])
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("Clusters – Projection PCA 2D", fontweight="bold"); ax.legend(markerscale=2)
        plt.tight_layout(); st.pyplot(fig)

        df_clu_r = df_clu.copy(); df_clu_r["Cluster"] = labels
        profil = df_clu_r.groupby("Cluster")[["duree_totale","nb_agents","Experience"]].mean().round(1)
        profil["Nb dossiers"] = df_clu_r["Cluster"].value_counts().sort_index()
        st.dataframe(profil, use_container_width=True)
        st.markdown('<div class="insight"> Clusters correspondant à des niveaux de complexité différents. Dossiers simples (court, peu d\'agents) vs complexes (long, multi-agents, forte expérience requise).</div>', unsafe_allow_html=True)

# 
#  PAGE 5 – SIMULATEUR
# 
elif page == "Simulateur":
    st.markdown('<div class="main-title">Simulateur – Prédiction interactive</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Entrez les caractéristiques d\'un dossier pour obtenir une prédiction</div>', unsafe_allow_html=True)
    if not DATA_OK: st.error("Données non disponibles."); st.stop()

    df_enc, label_encoders = build_ml_base(df_dossiers, df_temps, df_ressources)

    st.markdown('<div class="section-hdr"> Caractéristiques du dossier</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        cause   = st.selectbox("Cause d'intervention",  sorted(df_dossiers["Cause.intervention"].dropna().unique().tolist()))
        energie = st.selectbox("Type d'énergie",         sorted(df_dossiers["Type.d.energie"].dropna().unique().tolist()))
        outil   = st.selectbox("Outil d'assistance",     sorted(df_dossiers["Outil.d.assistance"].dropna().unique().tolist()))
    with c2:
        mois_sel  = st.slider("Mois d'ouverture", 1, 12, 6)
        annee_sel = st.selectbox("Année", [2021, 2022])
        top_dr    = st.selectbox("TOP D/R (dépannage)", [1, 0], format_func=lambda x: "Oui" if x else "Non")
    with c3:
        nb_agents   = st.slider("Nb agents sur le dossier", 1, 10, 2)
        duree_moy_a = st.slider("Durée moyenne par agent (s)", 30, 2000, 300)
        experience  = st.slider("Expérience de l'agent (jours)", 0, 8000, 2000)
        population  = st.selectbox("Type d'agent", ["CAS","CAC"])
        lieu        = st.selectbox("Lieu de travail", ["SITE","TELE"])

    duree_totale_in = nb_agents * duree_moy_a

    def enc(val, col):
        le = label_encoders.get(col)
        if le is None: return 0
        try: return int(le.transform([str(val)])[0])
        except: return 0

    feat_base = {
        "Cause.intervention":         enc(cause,  "Cause.intervention"),
        "Type.d.energie":             enc(energie,"Type.d.energie"),
        "Outil.d.assistance":         enc(outil,  "Outil.d.assistance"),
        "mois": mois_sel, "annee": annee_sel,
        "duree_totale": duree_totale_in, "nb_agents": nb_agents,
        "duree_moyenne": duree_moy_a, "Experience": experience,
        "Population":   enc(population,"Population"),
        "Lieu.travail": enc(lieu,"Lieu.travail"),
        "TOP.D.R": top_dr,
        "TOP.Rappat.valide":0,"TOP.Poursuite":0,"TOP.Autres.Garanties":0,
    }
    for col in ["Client","Formule","Assistance.ou.Administratif","Type.de.contrat"]:
        if col in df_enc.columns: feat_base[col] = int(df_enc[col].mode()[0])

    st.markdown("---")
    if st.button(" Lancer la prédiction", type="primary"):
        FEAT_CLF = ["Client","Formule","Cause.intervention","Type.d.energie",
                    "Outil.d.assistance","Assistance.ou.Administratif",
                    "mois","annee","duree_totale","nb_agents","duree_moyenne",
                    "Experience","Population","Type.de.contrat","Lieu.travail",
                    "TOP.D.R","TOP.Rappat.valide","TOP.Poursuite","TOP.Autres.Garanties"]
        FEAT_REG  = ["Client","Formule","Cause.intervention","Type.d.energie",
                     "Outil.d.assistance","Assistance.ou.Administratif",
                     "mois","annee","nb_agents","Experience","Population","Type.de.contrat","Lieu.travail",
                     "TOP.D.R","TOP.VR","TOP.Rappat.valide","TOP.Poursuite","TOP.Autres.Garanties"]

        feat_clf = [f for f in FEAT_CLF if f in df_enc.columns]
        feat_reg = [f for f in FEAT_REG  if f in df_enc.columns]

        if "TOP.VR" not in df_enc.columns or "duree_totale" not in df_enc.columns:
            st.error("Colonnes cibles manquantes."); st.stop()

        with st.spinner("Entraînement en cours..."):
            X_clf, y_clf = df_enc[feat_clf], df_enc["TOP.VR"]
            Xtr, Xte, ytr, yte = train_test_split(X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf)
            rf_c = RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)
            rf_c.fit(Xtr, ytr)

            df_rg = df_enc[feat_reg + ["duree_totale"]].dropna(subset=["duree_totale"])
            X_rg  = df_rg[feat_reg]; y_rg = np.log1p(df_rg["duree_totale"])
            Xtr2, _, ytr2, _ = train_test_split(X_rg, y_rg, test_size=0.25, random_state=42)
            gb_r = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
            gb_r.fit(Xtr2, ytr2)

        inp_clf   = pd.DataFrame([{f: feat_base.get(f,0) for f in feat_clf}])
        proba_vr  = rf_c.predict_proba(inp_clf)[0][1]
        pred_vr   = int(proba_vr >= 0.5)
        feat_base["TOP.VR"] = pred_vr
        inp_reg   = pd.DataFrame([{f: feat_base.get(f,0) for f in feat_reg}])
        pred_dur  = int(np.expm1(gb_r.predict(inp_reg)[0]))

        st.markdown('<div class="section-hdr"> Résultats</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            col_ = "#e63946" if pred_vr else "#2a9d8f"
            lbl  = "VR déclenché " if pred_vr else "Pas de VR "
            st.markdown(f'<div class="metric-card" style="background:linear-gradient(135deg,{col_},#1a1a2e)"><h2>{"OUI" if pred_vr else "NON"}</h2><p>{lbl}</p><p style="color:#fff;margin-top:8px">Probabilité : <b>{proba_vr*100:.1f}%</b></p></div>', unsafe_allow_html=True)
        with r2:
            mins = pred_dur // 60; secs = pred_dur % 60
            st.markdown(f'<div class="metric-card"><h2>{mins}m {secs}s</h2><p>Durée totale estimée</p><p style="color:#adb5bd;margin-top:8px">({pred_dur:,} secondes)</p></div>', unsafe_allow_html=True)
