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
nb_total_annees = len(df)
nb_annees_extremes = df['Est_Extreme'].sum()
nb_successifs = df['Successif_2_Ans'].sum()

prob_extreme = nb_annees_extremes / nb_total_annees
prob_conditionnelle = (nb_successifs / df['Extreme_Precedent'].sum()) if df['Extreme_Precedent'].sum() > 0 else 0.0

# --- NOUVEAU : CALCULS PRÉDICTIFS ANTICIPÉS ---
# 1. Autocorrélation
autocorr_val = df['Valeur_Analyse'].autocorr(lag=1)
seuil_significativite = 1.96 / np.sqrt(nb_total_annees)

# 2. Wald-Wolfowitz (Runs et Z-score)
runs = 1
for i in range(1, nb_total_annees):
    if df['Est_Extreme'].iloc[i] != df['Est_Extreme'].iloc[i-1]:
        runs += 1
n1 = nb_annees_extremes
n2 = nb_total_annees - n1
n = nb_total_annees
esp_runs = ((2 * n1 * n2) / n) if n > 0 else 1
esp_runs += 1
var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / ((n ** 2) * (n - 1)) if n > 1 else 1
std_runs = np.sqrt(var_runs) if var_runs > 0 else 1
z_score = (runs - esp_runs) / std_runs

# --- INDICATEURS CLÉS ---
col1, col2, col3 = st.columns(3)
col1.metric(f"Seuil d'alerte", f"{threshold_value:.1f} {unite}")
col2.metric("Années critiques", f"{nb_annees_extremes} / {nb_total_annees}")
col3.metric("Récurrences (2 ans de suite)", f"{nb_successifs}")

# --- NOUVEAU : SYNTHÈSE DYNAMIQUE INTELLIGENTE ---
# Z-score négatif = Les événements sont regroupés (Clustering). Z-score positif = Ils s'alternent trop.
if autocorr_val > seuil_significativite or z_score < -1.96:
    message_synthese = (f"💡 **Synthèse des Risques :** Historiquement, une année a **{prob_extreme * 100:.1f}%** de chances d'être critique. "
                        f"L'analyse mathématique prouve que ce bassin possède une **mémoire hydrologique** (les crues ont tendance à s'attirer). "
                        f"Puisque {prob_conditionnelle * 100:.1f}% des crises ont été consécutives par le passé, cette statistique ne doit pas être ignorée. "
                        f"Si l'année en cours est critique, le risque prédictif d'en subir une nouvelle l'an prochain reste très élevé (**~ {prob_conditionnelle * 100:.1f}%**).")

elif autocorr_val < -seuil_significativite or z_score > 1.96:
    message_synthese = (f"💡 **Synthèse des Risques :** Historiquement, une année a **{prob_extreme * 100:.1f}%** de chances d'être critique. "
                        f"Les tests statistiques montrent un phénomène d'**alternance** sur ce bassin : une crise est rarement suivie d'une autre crise. "
                        f"Le risque de subir une crue consécutive l'an prochain est donc statistiquement très faible, bien en dessous de sa probabilité normale.")

else:
    message_synthese = (f"💡 **Synthèse des Risques :** Historiquement, une année a **{prob_extreme * 100:.1f}%** de chances d'être critique. "
                        f"Si l'on a pu observer par le passé que {prob_conditionnelle * 100:.1f}% de ces crises ont été consécutives, les tests formels démontrent qu'il s'agit d'une simple coïncidence. "
                        f"Les années sont indépendantes. La probabilité d'avoir une crue l'an prochain redescend à sa normale statistique de **{prob_extreme * 100:.1f}%**.")

st.info(message_synthese)
st.write("---")
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
        
        # Calcul du seuil de significativité statistique (Intervalle de confiance à 95%)
        seuil_significativite = 1.96 / np.sqrt(nb_total_annees)
        
        # Génération dynamique de la conclusion selon la valeur
        if autocorr_val > seuil_significativite:
            interpretation = (f"Le coefficient ({autocorr_val:.2f}) est positif et dépasse le seuil critique (+{seuil_significativite:.2f}). "
                              "Cela prouve qu'il y a une **mémoire du système** : une année de crue a effectivement tendance à être suivie d'une autre année humide (effet de persistance, nappes phréatiques saturées, etc.). "
                              "Dans ce cas précis, la crainte de la direction est justifiée : le risque de récurrence (40%) a une réalité physique et doit être pris au sérieux.")
        elif autocorr_val < -seuil_significativite:
            interpretation = (f"Le coefficient ({autocorr_val:.2f}) est négatif et dépasse le seuil critique (-{seuil_significativite:.2f}). "
                              "Cela prouve une **alternance cyclique** : le système se compense (une crise est souvent suivie d'une année calme/sèche). "
                              "Le risque d'avoir deux crues d'affilée est donc mathématiquement bien plus faible que la normale.")
        else:
            interpretation = (f"Le coefficient ({autocorr_val:.2f}) reste à l'intérieur de la zone de hasard (entre -{seuil_significativite:.2f} et +{seuil_significativite:.2f}). "
                              "Cela indique qu'il n'y a **aucune corrélation statistique forte** d'une année sur l'autre. "
                              "Les crises successives passées sont des coïncidences (événements indépendants). Le risque pour l'année prochaine reste égal à la probabilité de base.")

        # Affichage du bloc avec les références
        st.info(f"**Coefficient d'autocorrélation (à 1 an) : {autocorr_val:.2f}**\n\n"
                f"📊 **Valeurs de référence (Seuil d'alerte pour {nb_total_annees} ans : ±{seuil_significativite:.2f}) :**\n"
                f"- **Autour de 0 ( < {seuil_significativite:.2f} )** : Indépendance totale (hasard pur).\n"
                f"- **Positif ( > +{seuil_significativite:.2f} )** : Effet de persistance (les crises s'attirent).\n"
                f"- **Négatif ( < -{seuil_significativite:.2f} )** : Effet d'alternance (les crises repoussent les crises).\n\n"
                f"💡 **Conclusion pour la direction :** {interpretation}")
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
        
        st.success("💡 **Comment fonctionne la Loi de Weibull (Périodes de Retour) :**\n\n"
                   "La formule de Weibull est le standard international en hydrologie pour évaluer la rareté d'une crue. Voici sa logique en 3 étapes :\n\n"
                   "1. **Le Classement :** L'algorithme prend notre historique complet et classe chaque année de la plus extrême (rang $m = 1$) à la plus faible.\n"
                   "2. **Le Calcul :** Il détermine la Période de Retour théorique ($T$) via la formule **$T = \\frac{N + 1}{m}$** (où $N$ est le nombre total d'années d'observation).\n"
                   "3. **Le Risque Annuel :** L'inverse de cette période ($1/T$) nous donne la véritable probabilité annuelle de l'événement.\n\n"
                   "**Conclusion pour la direction :** L'hydrologie n'a pas de 'mémoire'. Si le tableau indique qu'une crue passée est un événement décennal (Période = 10 ans), cela ne signifie pas qu'il faut attendre 10 ans pour la revoir, mais qu'elle a **exactement 10% de probabilité de se produire à chaque nouvelle année**, indépendamment de ce qui s'est passé hier. Ce tableau vous fournit donc le véritable 'risque fondamental' pour chaque seuil de gravité.")
        st.write("---")
    st.write("---")
    st.markdown("### 3. Test de Hasard des Séquences (Test de Wald-Wolfowitz)")
    st.caption("Évalue mathématiquement si l'apparition de crises consécutives (le phénomène des '2 ans') est une anomalie climatique ou une simple coïncidence statistique.")
    st.info("💡 **Le fonctionnement mathématique du test (L'analogie de la pièce) :**\n\n"
            "Imaginez que chaque année, la nature lance une pièce de monnaie (*Pile* = Année Normale, *Face* = Crise). "
            "Même avec une pièce parfaitement équilibrée, il arrivera de faire *Face* deux fois de suite par pur hasard.\n\n"
            "Le test de Wald-Wolfowitz compte le nombre réel d'alternances ($R$) entre ces deux états. "
            "Pour savoir si ce nombre est purement aléatoire, l'algorithme calcule d'abord l'espérance mathématique, c'est-à-dire le nombre théorique d'alternances attendu :\n\n"
            "$$E(R) = \\frac{2 n_1 n_2}{n} + 1$$\n\n"
            "*(Où $n_1$ est le nombre d'années critiques, $n_2$ le nombre d'années normales, et $n$ le nombre total d'années).* \n\n"
            "Il détermine ensuite la variance, qui représente la marge de fluctuation naturelle autorisée pour ce hasard :\n\n"
            "$$Var(R) = \\frac{2 n_1 n_2 (2 n_1 n_2 - n)}{n^2 (n - 1)}$$\n\n"
            "Enfin, on mesure l'écart exact entre notre historique et la théorie absolue grâce au Score Z :\n\n"
            "$$Z = \\frac{R - E(R)}{\\sqrt{Var(R)}}$$\n\n"
            "Si la valeur de $Z$ se situe entre -1.96 et 1.96, la théorie se confirme : avoir deux crises consécutives n'est pas le fruit d'un cycle climatique dangereux, mais simplement un 'double Face' accidentel.")
    
    # 1. Calcul du nombre de séquences (Runs) réelles
    runs = 1
    for i in range(1, nb_total_annees):
        if df['Est_Extreme'].iloc[i] != df['Est_Extreme'].iloc[i-1]:
            runs += 1
            
    n1 = nb_annees_extremes       # Années critiques
    n2 = nb_total_annees - n1     # Années normales
    n = nb_total_annees           # Total
    
    # 2. Calcul des valeurs théoriques (Loi Binomiale)
    esp_runs = ((2 * n1 * n2) / n) + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / ((n ** 2) * (n - 1))
    std_runs = np.sqrt(var_runs) if var_runs > 0 else 1
    
    # 3. Calcul du Z-Score
    z_score = (runs - esp_runs) / std_runs
    
    # 4. Affichage des métriques
    col_x, col_y, col_z = st.columns(3)
    col_x.metric("Alternances réelles", runs, help="Nombre de fois où l'on est passé d'une année normale à extrême, ou inversement.")
    col_y.metric("Alternances théoriques (Hasard)", f"{esp_runs:.1f}", help="Ce que les mathématiques prévoient si les années sont 100% indépendantes.")
    col_z.metric("Score Z (Écart)", f"{z_score:.2f}", help="S'il est entre -1.96 et 1.96, les événements sont prouvés comme étant aléatoires.")
    
    # --- 5. NOUVEAU : GRAPHIQUE D'ILLUSTRATION DU TEST ---
    fig3, ax3 = plt.subplots(figsize=(10, 3.5))
    
    # Création de la courbe en cloche (distribution théorique)
    x = np.linspace(esp_runs - 4*std_runs, esp_runs + 4*std_runs, 500)
    y = (1 / (std_runs * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - esp_runs) / std_runs)**2)
    ax3.plot(x, y, color='#94A3B8', linewidth=2)
    
    # Zone de confiance (Hasard validé)
    x_fill = np.linspace(esp_runs - 1.96*std_runs, esp_runs + 1.96*std_runs, 500)
    y_fill = (1 / (std_runs * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fill - esp_runs) / std_runs)**2)
    ax3.fill_between(x_fill, y_fill, color='#E0F2FE', alpha=0.8, label="Zone de pur hasard statistique")
    
    # Placement de l'historique réel
    ax3.axvline(x=esp_runs, color='#94A3B8', linestyle=':', linewidth=1.5, label=f"Moyenne théorique ({esp_runs:.1f})")
    ax3.axvline(x=runs, color='#EF4444', linestyle='-', linewidth=3, label=f"Notre historique ({runs} alternances)")
    
    ax3.set_yticks([]) # On cache l'axe Y car la densité n'est pas pertinente pour un public non technique
    ax3.set_xlabel("Nombre d'alternances", color='#475569')
    ax3.tick_params(colors='#475569')
    ax3.legend(loc='upper right', facecolor='#FFFFFF', edgecolor='#E2E8F0')
    sns.despine(left=True)
    
    st.pyplot(fig3)
    # -----------------------------------------------------

    # 6. Conclusion dynamique pour la direction
    if abs(z_score) < 1.96:
        st.success(f"✅ **Démonstration Mathématique :** Notre historique se situe parfaitement dans la zone bleue (Score Z = {z_score:.2f}). \n\n"
                   f"**Ce que cela signifie pour la direction :** Le fait que plusieurs crises se soient produites 2 ans de suite par le passé relève d'une stricte **coïncidence probabiliste** et non d'un cycle hydrologique. Les crues sont des événements 100% indépendants. Il est donc mathématiquement erroné d'utiliser le chiffre de 40% pour prédire l'année prochaine. Le véritable risque de récurrence l'année prochaine est égal à la probabilité de base.")
    else:
        st.warning(f"⚠️ **Démonstration Mathématique :** Notre historique sort de la zone bleue du hasard (Score Z = {z_score:.2f}). \n\n"
                   "**Ce que cela signifie pour la direction :** Le test prouve qu'il y a bien un phénomène de regroupement ('clustering') non-aléatoire sur ce bassin. Les années humides ont tendance à s'enchaîner anormalement. La crainte de la direction est justifiée : le risque de récurrence (40%) doit être pris au sérieux pour l'année prochaine.")
