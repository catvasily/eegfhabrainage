import sys
import numpy as np
import sqlite3
from sqlite3 import Error as dbError

class ReportsDB:
    """
        A wrapper class for communicating with an existing reports database.

        **Attributes**

        Attributes:
            conn: the db connection object
            cursor: the db cursors object

        **Methods**
    """
    REPORTS = 'reports'     # The name of the main table

    # Column names in the REPORTS table
    HASH = 'Hashed ID'
    DOCTOR = 'Physician'
    HOSPITAL = 'Hospital'
    SCAN_ID = 'Scan ID'
    REP = 'Report'
    NOTE = 'Note'

    def __init__(self, db_file):
        """
        Constructor. Initializes the DB connection.

        Args:
            db_file (str): full pathname to the database file
        """
        self.conn = None
        self.cursor = None

        try:
            self.conn = sqlite3.connect(db_file)
        except dbError as e:
            print(e)
            sys.exit()

        self.cursor = self.conn.cursor()

    def __del__(self):
        """
        Destructor
        """
        if self.conn is not None:
            self.conn.close()

    def execute(self, sql, parms = ()):
        """
        Execute a SQL statement on current cursor.

        Args:
            sql (str): the SQL command to run
            parms (dict | list): optional arguments to the SQL command;
                a `dict` if named placeholders are used, a list if unnamed.

        Returns:
            rows (list of sqlite3.Row): A list of resulting rows (if any)
        """
        self.cursor.execute(sql, parms)
        return self.cursor.fetchall()

    def list_hospitals(self):
        """
        Return a list of all hospitals in DB
        """
        sql = f'SELECT DISTINCT {self.HOSPITAL} FROM {self.REPORTS};'
        rows = self.execute(sql)
        return [r[0] for r in rows]

    def list_physicians(self):
        """
        Return a list of all hospitals in DB
        """
        sql = f'SELECT DISTINCT {self.DOCTOR} FROM {self.REPORTS};'
        rows = self.execute(sql)
        return [r[0] for r in rows]

    def get_all_ids(self):
        """
        Return a list of ALL report IDs in the database
        """
        # Using DISTINCT here because there may be dupes working copies of DB
        sql = f'SELECT DISTINCT "{self.HASH}" FROM {self.REPORTS};'
        rows = self.execute(sql)
        return [r[0] for r in rows]

    def get_ids_for_physician(self, name):
        """
        Return a list of report IDs for the specified physician
        """
        # Using DISTINCT here because there may be dupes working copies of DB
        sql = f'SELECT DISTINCT "{self.HASH}" FROM {self.REPORTS} WHERE {self.DOCTOR}="{name}";'
        rows = self.execute(sql)
        return [r[0] for r in rows]

    def get_ids_for_hospital(self, name):
        """
        Return a list of report IDs for the specified hospital
        """
        # Using DISTINCT here because there may be dupes working copies of DB
        sql = f'SELECT DISTINCT "{self.HASH}" FROM {self.REPORTS} WHERE {self.HOSPITAL}="{name}";'
        rows = self.execute(sql)
        return [r[0] for r in rows]

    def get_reps_by_ids(self, lst_ids):
        """
        Retrieve texts of reports for a list of report IDs.

        IMPORTANT: Generally, the order of returned reports is DIFFERENT
        from the order of IDs in `lst_ids`.

        NOTE: For a very long `lst_ids` list this function may be inefficient, or even
        return an error (for tens of thousands of IDs). If a list of IDs is created
        using one of  `get_ids_for_` methods, **it is better to use corresponding `get_reps_for_...`
        methods** to retrieve reports.

        Args:
            lst_ids(sequence of str): a list of hashed IDs of the reports.

        Returns:
            dict_reps(dict): a dictionary mapping hashed ID -> report (a text string)
        """
        # TO DO: ?? Make it possible to specify only start of the hash

        # Using a straightforward approach with a list of IDs in a single SELECT with
        # 'WHERE .. IN' clause. However this may be inefficient and cause problems
        # for lists of thousands of IDs.

        str_lst = str(list(lst_ids))[1:-1].replace('\'', '"')
        
        sql = f'SELECT "{self.HASH}", "{self.REP}" FROM {self.REPORTS} '
        sql += f'WHERE "{self.HASH}" IN ({str_lst});'
        
        # NOTE: rows are not necessarily returned in the order of lst_ids
        rows = self.execute(sql)
        return {h:r for h, r in rows}

    def get_all_reps(self):
        """
        Return ALL reports from a database as a dictionary {'Hashed ID': report}
        """
        # Using DISTINCT here because there may be dupes working copies of DB
        sql = f'SELECT DISTINCT "{self.HASH}", "{self.REP}" FROM {self.REPORTS};'
        rows = self.execute(sql)
        return {h:r for h, r in rows}

    def get_reps_for_physician(self, name):
        """
        Return a dictionary {'Hashed ID': report} for the specified physician
        """
        # Using DISTINCT here because there may be dupes working copies of DB
        sql = f'SELECT DISTINCT "{self.HASH}", "{self.REP}" FROM {self.REPORTS} WHERE {self.DOCTOR}="{name}";'
        rows = self.execute(sql)
        return {h:r for h, r in rows}

    def get_reps_for_hospital(self, name):
        """
        Return a dictionary {'Hashed ID': report} for the specified hospital
        """
        # Using DISTINCT here because there may be dupes working copies of DB
        sql = f'SELECT DISTINCT "{self.HASH}", "{self.REP}" FROM {self.REPORTS} WHERE {self.HOSPITAL}="{name}";'
        rows = self.execute(sql)
        return {h:r for h, r in rows}

