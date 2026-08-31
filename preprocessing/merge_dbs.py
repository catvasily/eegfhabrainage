"""
Utility script to create a single SQLite DB that matches EEG scan Ids
with corresponding report IDs, hospitals, physicians and LLM-generated
labels.
"""

import sqlite3
import os

# Define database file paths (update these paths as needed)
DB_ROOT = '/data/eegfhabrainage/clean_eeg_reps'                     # DBs root folder
EEG_INFO_DB = 'eeg_release_001_main_public_250819.db'               # ScanID -> Hashed_ReportURN
REPORTS_DB = 'eeg_reports_release_001_aux_public_240325_Others.db'  # Hashed ID (report), Physician, Hospital
LLM_LABELS_DB = 'eeg_reports_release_001_mistral_public_250825.db'  # Hashed_ReportURN -> LLM classifications
MERGED_DB = 'eeg2rep2LLMlabels260705.db'                            # Merged from the three DBs above

def merge_eeg_databases():
    # Remove existing DB4 if it exists to ensure a clean run
    if os.path.exists(MERGED_DB):
        os.remove(MERGED_DB)
        print(f"Removed existing {MERGED_DB}")

    # Connect to the new destination database (DB4)
    conn = sqlite3.connect(MERGED_DB)
    cursor = conn.cursor()

    try:
        print("Attaching source databases...")
        # Attach the three source databases to the current connection
        cursor.execute(f"ATTACH DATABASE '{EEG_INFO_DB}' AS db1")
        cursor.execute(f"ATTACH DATABASE '{REPORTS_DB}' AS db2")
        cursor.execute(f"ATTACH DATABASE '{LLM_LABELS_DB}' AS db3")

        print("Fetching columns from DB2 to exclude 'Hashed ID'...")
        # Get column names from DB2's 'reports'
        cursor.execute("PRAGMA db2.table_info('reports')")
        db2_columns = [row[1] for row in cursor.fetchall()]
        db2_columns_clean = [f"db2.reports.`{col}`" for col in db2_columns if col != 'Hashed ID']
        db2_columns_str = ", ".join(db2_columns_clean)

        print("Fetching columns from DB3 to exclude 'Hashed_ReportURN'...")
        # Get column names from DB3's 'classifications'
        cursor.execute("PRAGMA db3.table_info('classifications')")
        db3_columns = [row[1] for row in cursor.fetchall()]
        # Exclude 'Hashed_ReportURN' to prevent duplication
        db3_columns_clean = [f"db3.classifications.`{col}`" for col in db3_columns if col != 'Hashed_ReportURN']
        db3_columns_str = ", ".join(db3_columns_clean)

        print("Creating table and inserting merged data...")
        # Construct the SQL query with clean, explicit columns for DB2 and DB3
        # This version allows dupes in Hashed_ReportURN:
        #query = f"""
        #CREATE TABLE eeg_classifications AS
        #SELECT 
        #    db1.`EEG metadata`.ScanID, 
        #    db1.`EEG metadata`.Hashed_ReportURN,
        #    {db2_columns_str},
        #    {db3_columns_str}
        #FROM db1.`EEG metadata`
        #INNER JOIN db2.reports 
        #    ON db1.`EEG metadata`.Hashed_ReportURN = db2.reports.`Hashed ID`
        #INNER JOIN db3.classifications 
        #    ON db1.`EEG metadata`.Hashed_ReportURN = db3.classifications.Hashed_ReportURN
        #"""

        # This version allows discards all records where report URN was used more than once:
        query = f"""
        CREATE TABLE eeg_classifications AS
        SELECT 
            db1.`EEG metadata`.ScanID, 
            db1.`EEG metadata`.Hashed_ReportURN,
            {db2_columns_str},
            {db3_columns_str}
        FROM db1.`EEG metadata`
        INNER JOIN db2.reports 
            ON db1.`EEG metadata`.Hashed_ReportURN = db2.reports.`Hashed ID`
        INNER JOIN db3.classifications 
            ON db1.`EEG metadata`.Hashed_ReportURN = db3.classifications.Hashed_ReportURN
        WHERE db1.`EEG metadata`.Hashed_ReportURN IN (
            SELECT Hashed_ReportURN
            FROM db1.`EEG metadata`
            GROUP BY Hashed_ReportURN
            HAVING COUNT(*) = 1
        );
        """
        cursor.execute(query)
        conn.commit()
        print(f"Successfully created {MERGED_DB} with table 'eeg_classifications' (no duplicates).")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        conn.rollback()

    finally:
        # Detach databases and close the connection safely
        try:
            cursor.execute("DETACH DATABASE db1")
            cursor.execute("DETACH DATABASE db2")
            cursor.execute("DETACH DATABASE db3")
        except sqlite3.Error:
            pass 
        
        conn.close()

if __name__ == "__main__":
    EEG_INFO_DB = DB_ROOT + '/' + EEG_INFO_DB 
    REPORTS_DB = DB_ROOT + '/' + REPORTS_DB 
    LLM_LABELS_DB = DB_ROOT + '/' + LLM_LABELS_DB 
    MERGED_DB = DB_ROOT + '/' + MERGED_DB 

    merge_eeg_databases()
    
