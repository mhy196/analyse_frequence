import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Tableau de Bord Hydrologique", layout="wide", initial_sidebar_state="expanded")

# --- DESIGN & CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
    h1, h2, h3 { color: #0F172A !important; font-family: 'Helvetica Neue', sans-serif; }
    [data-testid="stMetricLabel"] p { color: #475569 !important; font-size: 16px !important; font-weight: bold !important; visibility: visible !important; }
    [data-testid="stMetricValue"] { color: #0284C7; font-weight: bold; }
    button[data-baseweb="tab"] * { color: #0F172A !important; font-size: 17px !important; font-weight: 600 !important; }
    div.stAlert { background-color: #E0F2FE; border: 1px solid #BAE6FD; color: #0369A1; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- BARRE LATÉRALE (CHOIX DE LA STATION ET PARAMÈTRES) ---
st.sidebar.header("⚙️ Paramétrage")

station_choisie = st.sidebar.selectbox(
    "📍 Station hydrologique étudiée",
    options=["Aïn Aïcha", "Azib Soltane"]
)

st.sidebar.write("Ajustez les critères ci-dessous pour actualiser l'analyse.")

percentile_threshold = st.sidebar.slider(
    "Seuil d'Alerte (Percentile)",
    min_value=50, max_value=99, value=90, step=1,
    help="Détermine la rareté de l'événement. À 90, on isole les 10% des pires années."
)

aggregation_method = st.sidebar.selectbox(
    "Méthode d'évaluation de l'année",
    options=["Moyenne annuelle (Module)", "Maximum mensuel absolu"],
    help="Analyser l'année sur sa moyenne globale ou sur son mois le plus extrême."
)

# --- TITRE DYNAMIQUE ---
st.title(f"🌊 Suivi des Crises Hydrologiques ({station_choisie})")
st.markdown("Outil d'aide à la décision : Analyse de la fréquence et de la récurrence des événements extrêmes.")
st.write("---")

# --- CHARGEMENT DES DONNÉES (DYNAMIQUE) ---
@st.cache_data
def load_data(fichier):
    df = pd.read_excel(fichier, sheet_name='qm', header=6)
    df = df.dropna(subset=['Année'])
    mois = ['Septembre', 'Octobre', 'Novembre', 'Décembre', 'Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août']
    df[mois] = df[mois].fillna(0) 
    return df, mois

# Sélection du fichier en fonction de la station
if station_choisie == "Aïn Aïcha":
    fichier_excel = "Ain Aisha.xls"
else:
    fichier_excel = "Azib Soltane.xls"

try:
    df, mois = load_data(fichier_excel)
except Exception as e:
    st.error(f"Erreur lors du chargement du fichier {fichier_excel} : {e}")
    st.info("💡 Assurez-vous que le fichier est bien présent dans le dossier sur GitHub.")
    st.stop()

# --- LOGIQUE DYNAMIQUE (UNITÉS ET CALCULS) ---
if aggregation_method == "Moyenne annuelle (Module)":
    # On calcule la moyenne et on convertit en Apport Annuel (Millions de m3)
    df['Valeur_Analyse'] = df[mois].mean(axis=1) * 31.536
    unite = "Mm³"
    nom_variable = "Apport Annuel"
else:
    # On isole le pic mensuel en le gardant en m3/s (débit de pointe)
    df['Valeur_Analyse'] = df[mois].max(axis=1)
    unite = "m³/s"
    nom_variable = "Débit Pic Mensuel"

# --- TRAITEMENT STATISTIQUE ---
threshold_value = np.percentile(df['Valeur_Analyse'], percentile_threshold)
df['Est_Extreme'] = df['Valeur_Analyse'] >= threshold_value
df['Extreme_Precedent'] = df['Est_Extreme'].shift(1, fill_value=False)
df['Successif_2_Ans'] = df['Est_Extreme'] & df['Extreme_Precedent']

nb_total_annees = len(df)
nb_annees_extremes = df['Est_Extreme'].sum()
nb_successifs = df['Successif_2_Ans'].sum()

prob_extreme = nb_annees_extremes / nb_total_annees
prob_conditionnelle = (nb_successifs / df['Extreme_Precedent'].sum()) if df['Extreme_Precedent'].sum() > 0 else 0.0

# --- INDICATEURS CLÉS ---
col1, col2, col3 = st.columns(3)
col1.metric(f"Seuil d'alerte", f"{threshold_value:.1f} {unite}")
col2.metric("Années critiques", f"{nb_annees_extremes} / {nb_total_annees}")
col3.metric("Récurrences (2 ans de suite)", f"{nb_successifs}")

st.info(f"💡 **Conclusion Statistique :** Il y a **{prob_extreme * 100:.1f}%** de chances d'avoir une année critique de manière globale. Toutefois, si une année critique survient, le risque de subir une seconde année critique immédiatement après s'élève à **{prob_conditionnelle * 100:.1f}%**.")
st.write("---")

# --- ONGLETS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Chronologie", "📊 Répartition", "🗂️ Base de données", "📋 Statistiques descriptives", "🔮 Prédictions & Risques"])

sns.set_style("whitegrid")
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['figure.facecolor'] = 'none' 
plt.rcParams['grid.color'] = '#E2E8F0'

with tab1:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['Année'], df['Valeur_Analyse'], marker='o', linestyle='-', color='#0284C7', label=f"{nom_variable} ({unite})", linewidth=1.5)
    ax.axhline(y=threshold_value, color='#EF4444', linestyle='--', linewidth=2, label='Seuil critique')
    
    extremes = df[df['Est_Extreme']]
    ax.scatter(extremes['Année'], extremes['Valeur_Analyse'], color='#EF4444', s=80, zorder=5, label='Année Critique Isolée')
    
    successifs = df[df['Successif_2_Ans']]
    ax.scatter(successifs['Année'], successifs['Valeur_Analyse'], color='#F59E0B', edgecolor='black', s=120, zorder=6, label='Crise Récurrente (2ème année)')
    
    plt.xticks(rotation=45)
    ax.set_ylabel(f"{nom_variable} ({unite})", color='#475569')
    ax.tick_params(colors='#475569')
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % 3 != 0: label.set_visible(False)
    ax.legend(facecolor='#FFFFFF', edgecolor='#E2E8F0')
    st.pyplot(fig)

with tab2:
    st.caption("Catégorisation des années selon leur déclenchement.")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    
    # NOUVELLE LOGIQUE : Plus simple et directement liée aux indicateurs
    conditions = [
        (~df['Est_Extreme']),                                     # 1. Sous le seuil
        (df['Est_Extreme'] & ~df['Extreme_Precedent']),           # 2. Première année de crise (isolée ou début)
        (df['Successif_2_Ans'])                                   # 3. 2ème année ou plus (La récurrence)
    ]
    choices = ['Année Normale', 'Nouvelle Crise (1ère année)', 'Crise Récurrente (Suite)']
    df['Categorie'] = np.select(conditions, choices, default='Année Normale')

    # Création du graphique avec les nouvelles couleurs
    sns.countplot(data=df, x='Categorie', order=choices, palette=['#BAE6FD', '#F59E0B', '#991B1B'], ax=ax2)
    
    ax2.set_ylabel("Nombre d'années", color='#475569')
    ax2.set_xlabel("")
    ax2.tick_params(colors='#475569')
    
    # Ajout des chiffres sur les barres
    for p in ax2.patches:
        ax2.annotate(format(p.get_height(), '.0f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', 
                     xytext = (0, 9), textcoords = 'offset points',
                     color='#1E293B', fontweight='bold')

    sns.despine(left=True, bottom=True)
    st.pyplot(fig2)

with tab3:
    df_display = df[['Année', 'Valeur_Analyse', 'Est_Extreme', 'Successif_2_Ans', 'Categorie']].copy()
    df_display['Est_Extreme'] = df_display['Est_Extreme'].map({True: 'Oui', False: 'Non'})
    df_display['Successif_2_Ans'] = df_display['Successif_2_Ans'].map({True: 'Oui', False: 'Non'})
    df_display['Valeur_Analyse'] = df_display['Valeur_Analyse'].round(2)
    df_display.columns = ['Année Hydrologique', f'{nom_variable} ({unite})', 'Dépasse le seuil ?', '2ème année de crise ?', 'Statut']
    st.dataframe(df_display, use_container_width=True)

with tab4:
    st.subheader(f"Statistiques globales ({unite})")
    
    val_mean = df['Valeur_Analyse'].mean()
    val_median = df['Valeur_Analyse'].median()
    val_max = df['Valeur_Analyse'].max()
    val_min = df['Valeur_Analyse'].min()
    val_std = df['Valeur_Analyse'].std()
    
    annee_max = df.loc[df['Valeur_Analyse'].idxmax(), 'Année']
    annee_min = df.loc[df['Valeur_Analyse'].idxmin(), 'Année']
    
    s_col1, s_col2 = st.columns(2)
    
    with s_col1:
        st.markdown(f"**Moyenne :** {val_mean:.2f} {unite}")
        st.markdown(f"**Médiane :** {val_median:.2f} {unite}")
        st.markdown(f"**Écart-type (Volatilité) :** {val_std:.2f} {unite}")
        
    with s_col2:
        st.markdown(f"**Maximum Historique :** {val_max:.2f} {unite} (enregistré en **{annee_max}**)")
        st.markdown(f"**Minimum Historique :** {val_min:.2f} {unite} (enregistré en **{annee_min}**)")
        st.markdown(f"**Nombre d'années observées :** {nb_total_annees} ans")

with tab5:
    st.subheader("Analyse Prédictive et Indépendance Statistique")
    st.write("Ces modèles hydrologiques permettent de vérifier si une crise influence réellement l'année suivante, afin d'éviter les biais d'interprétation sur la probabilité conditionnelle.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 1. Test d'Autocorrélation (Mémoire du système)")
        st.caption("Vérifie mathématiquement si l'année N influence l'année N+1.")
        
        # Génération du graphique d'autocorrélation avec Pandas
        fig_acf, ax_acf = plt.subplots(figsize=(6, 4))
        pd.plotting.autocorrelation_plot(df['Valeur_Analyse'], ax=ax_acf)
        ax_acf.set_title("Corrélation interannuelle")
        
        # Personnalisation des couleurs pour le thème clair
        ax_acf.tick_params(colors='#475569')
        for line in ax_acf.lines:
            line.set_color('#0284C7')
        sns.despine()
        
        st.pyplot(fig_acf)
        
        # Calcul du coefficient lag-1
        autocorr_val = df['Valeur_Analyse'].autocorr(lag=1)
        
        # Interprétation dynamique pour le boss
        st.info(f"**Coefficient d'autocorrélation (à 1 an) : {autocorr_val:.2f}**\n\n"
                f"💡 **Conclusion pour la direction :** Un coefficient proche de 0 indique qu'il n'y a **aucune corrélation statistique** entre l'apport d'une année et celui de la suivante. Les événements extrêmes successifs observés dans le passé sont donc des coïncidences statistiques (indépendance des événements) et non une règle prédictive. Le risque réel pour l'année prochaine reste égal à la probabilité de base.")

    with col_b:
        st.markdown("### 2. Périodes de Retour (Loi de Weibull)")
        st.caption("Classement des événements historiques pour déterminer leur véritable probabilité annuelle théorique.")
        
        # Calcul de la période de retour empirique (formule de Weibull : T = (N+1)/m)
        df_sorted = df[['Année', 'Valeur_Analyse']].sort_values(by='Valeur_Analyse', ascending=False).reset_index(drop=True)
        df_sorted['Rang (m)'] = df_sorted.index + 1
        N = len(df_sorted)
        df_sorted['Période de Retour (T)'] = (N + 1) / df_sorted['Rang (m)']
        df_sorted['Probabilité Annuelle'] = (1 / df_sorted['Période de Retour (T)']) * 100
        
        # Formatage du tableau pour l'affichage
        df_display_rp = df_sorted.head(10).copy()
        df_display_rp['Valeur_Analyse'] = df_display_rp['Valeur_Analyse'].round(1)
        df_display_rp['Période de Retour (T)'] = df_display_rp['Période de Retour (T)'].round(1).astype(str) + " ans"
        df_display_rp['Probabilité Annuelle'] = df_display_rp['Probabilité Annuelle'].round(1).astype(str) + " %"
        
        df_display_rp = df_display_rp[['Année', 'Valeur_Analyse', 'Période de Retour (T)', 'Probabilité Annuelle']]
        df_display_rp.columns = ['Année', f'{nom_variable} ({unite})', 'Période de Retour (T)', 'Risque Annuel (1/T)']
        
        st.dataframe(df_display_rp, use_container_width=True)
        
        st.success("💡 **Explication des Périodes de Retour :** En hydrologie de crue, la probabilité annuelle est fixe. Si une année extrême a un 'Risque Annuel' de 10%, alors peu importe s'il y a eu une crue cette année, les chances mathématiques d'avoir une crue équivalente l'année prochaine sont exactement de **10%**.")
