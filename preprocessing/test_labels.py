"""
Test if binary labels and their confidences are correct.
Specifically, that DB classifications 1, 2 map to binary label 0
with 'confident' flags 1,0, respectively, and that classifications
3, 4 in DB map to binary label 1 with confident flag 0, 1, respectively.

NOTE. One needs to use files 'predict_summary_...<label>...csv' with 
the summary rows at the end removed, and version of LLM classifications
DB where there is a table with name 'eeg_classifications'.
"""

import os
import sqlite3
import pandas as pd


def validate_mapping(df, db_column):
    """Helper function to apply the conditional logic to a dataframe.

    Validates that:
    1 -> TrueLabel=0, Confident=1
    2 -> TrueLabel=0, Confident=0
    3 -> TrueLabel=1, Confident=0
    4 -> TrueLabel=1, Confident=1
    """

    def check_row_logic(row):
        db_val = row[db_column]
        label = row["TrueLabel"]
        conf = row["Confident"]

        if db_val == 1:
            return label == 0 and conf == 1
        elif db_val == 2:
            return label == 0 and conf == 0
        elif db_val == 3:
            return label == 1 and conf == 0
        elif db_val == 4:
            return label == 1 and conf == 1
        return False

    # Apply the check to every row
    return df.apply(check_row_logic, axis=1)


def run_tests(csv1_path, csv2_path, db_path):
    # 1. Verification of files
    for path in [csv1_path, csv2_path, db_path]:
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            return

    # 2. Read common SQLite Database table
    try:
        conn = sqlite3.connect(db_path)
        #query = "SELECT ScanID, `Gen Non-epi`, Abnormality FROM others_gen_non_epi"
        query = "SELECT ScanID, `Gen Non-epi`, Abnormality FROM eeg_classifications"
        df_db = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print(f"Error reading SQLite DB: {e}")
        return

    csv_cols = ["ScanID", "TrueLabel", "Confident"]

    # ==========================================
    # TEST 1: qq_abnorm.csv vs Abnormality
    # ==========================================
    print("\n--- Running Test 1: qq_abnorm.csv vs Abnormality ---")
    try:
        df_csv1 = pd.read_csv(csv1_path, usecols=csv_cols)
        df_merged1 = pd.merge(df_csv1, df_db, on="ScanID", how="inner")

        if df_merged1.empty:
            print("Warning: No matching ScanIDs for Test 1.")
        else:
            df_merged1["Passed_Test"] = validate_mapping(
                df_merged1, "Abnormality"
            )
            failures1 = df_merged1[df_merged1["Passed_Test"] == False]

            if failures1.empty:
                print(
                    f"✅ SUCCESS: All {len(df_merged1)} matching ScanIDs passed."
                )
            else:
                print(
                    f"❌ FAILURE: Found {len(failures1)} mismatching record(s) out of {len(df_merged1)} total."
                )
                print(
                    failures1[
                        ["ScanID", "Abnormality", "TrueLabel", "Confident"]
                    ].to_string(index=False)
                )
    except Exception as e:
        print(f"Error running Test 1: {e}")

    # ==========================================
    # TEST 2: qq_gennonepi.csv vs Gen Non-epi
    # ==========================================
    print("\n--- Running Test 2: qq_gennonepi.csv vs Gen Non-epi ---")
    try:
        df_csv2 = pd.read_csv(csv2_path, usecols=csv_cols)
        df_merged2 = pd.merge(df_csv2, df_db, on="ScanID", how="inner")

        if df_merged2.empty:
            print("Warning: No matching ScanIDs for Test 2.")
        else:
            df_merged2["Passed_Test"] = validate_mapping(
                df_merged2, "Gen Non-epi"
            )
            failures2 = df_merged2[df_merged2["Passed_Test"] == False]

            if failures2.empty:
                print(
                    f"✅ SUCCESS: All {len(df_merged2)} matching ScanIDs passed."
                )
            else:
                print(
                    f"❌ FAILURE: Found {len(failures2)} mismatching record(s) out of {len(df_merged2)} total."
                )
                print(
                    failures2[
                        ["ScanID", "Gen Non-epi", "TrueLabel", "Confident"]
                    ].to_string(index=False)
                )
    except Exception as e:
        print(f"Error running Test 2: {e}")


if __name__ == "__main__":
    # File paths
    CSV1_FILE = "/data/eegfhabrainage/src-cls/physicians/ccv-260329/predict_F1/qq_abnorm.csv"
    CSV2_FILE = "/data/eegfhabrainage/src-cls/physicians/ccv-260329/predict_F1/qq_gennonepi.csv"
    DB_FILE = "/data/eegfhabrainage/clean_eeg_reps/work.db"  # <- Update to your actual DB file name

    run_tests(CSV1_FILE, CSV2_FILE, DB_FILE)

