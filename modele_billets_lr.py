import pandas as pd
import joblib

# Chemin vers le modèle sauvegardé
MODELE_PATH = "modele_billets_lr.joblib"

# Colonnes attendues
COLS = ["diagonal", "height_left", "height_right",
        "margin_low", "margin_up", "length"]

# Chargement du modèle (pipeline : imputer + scaler + régression logistique)
modele = joblib.load(MODELE_PATH)


def predict_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prédit la nature des billets à partir d'un DataFrame contenant
    les 6 dimensions géométriques.
    Retourne un DataFrame avec une colonne is_genuine_pred.
    """
    df_input = df[COLS].copy()
    preds = modele.predict(df_input)

    df_result = df.copy()
    df_result["is_genuine_pred"] = preds
    return df_result


def predict_from_csv(path_csv: str, sep: str = ";") -> pd.DataFrame:
    """
    Lit un fichier CSV et renvoie un DataFrame avec les prédictions.
    """
    df = pd.read_csv(path_csv, sep=sep)
    return predict_from_dataframe(df)


def predict_from_values(diagonal, height_left, height_right,
                        margin_low, margin_up, length):
    df = pd.DataFrame([{
        "diagonal": diagonal,
        "height_left": height_left,
        "height_right": height_right,
        "margin_low": margin_low,
        "margin_up": margin_up,
        "length": length
    }])

    df_input = df[COLS].copy()
    pred = modele.predict(df_input)[0]
    proba_vrai = modele.predict_proba(df_input)[0, 1]
    return pred, proba_vrai

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage : python modele_billets_lr.py <chemin_vers_fichier_csv>")
        sys.exit(1)

    # 1) chemin du fichier d'entrée (fourni en argument)
    input_path = sys.argv[1]

    # 2) prédictions à partir du CSV
    df_pred = predict_from_csv(input_path, sep=";")

    # 3) on enregistre les prédictions dans un nouveau CSV
    output_path = "predictions_billets.csv"
    df_pred.to_csv(output_path, index=False)

    print(f"Prédictions terminées. Fichier généré : {output_path}")