"""
Functions for creating a SQLite database and adding tables to it.
"""
import sys
import os.path as path
import sqlite3
from sqlite3 import Error

def create_sqlite_db(db_file, db_tables):
    """
    Initialize a SQLite database and create all necessary
    tables.

    If a database under the specified name already exists, this
    functions aborts.

    As a result a new SQLite database file with empty tables is
    created.

    Args:
        db_file (str): full pathname of the new database
        db_tables (dict): a dictionary with items that describe
            each database table to be created, in the form:   
            <table_name>: [..., [<Column-name>, Column-type>],...]
            
    Returns:
        nothing
    """
    if path.isfile(db_file):
        raise ValueError(f'Creating database failed: file {db_file} already exists')

    add_db_tables(db_file, db_tables)
    print(f'Successfully created database {db_file}, SQLite version {sqlite3.version}')

def add_db_tables(db_file, tbls, return_cursor = False):
    """
    Add new tables to a SQLite database.

    Args:
        db_file(str): full pathname of the DB file
        tbls (dict): a dictionary of table descriptors, each one being
            <table_name>: [..., [<Column-name>, Column-type>],...]
        return_cursor (bool): flag to keep database connection open and return
            the `Cursor` object. If `True`, it's the caller responsibility to
            close the database connection by executing `cursor.connection.close()`

    Returns:
        cursor or None: the cursor object is returned if `return_cursor` is `True`

    NOTE:
        When specifying multi-word strings for column names or table names, those
        should be placed inside double quotes. For example, for a column named 'Column 1'
        in a 'Table 1', corresponding key in the `tbls` dictionary should be   
        "\"Table 1\"": [["\"Column 1\""], "TEXT"],...,]
    """
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
        sys.exit()

    # Tables
    cursor = conn.cursor()

    for t in tbls:
        sql = f'CREATE TABLE {t} ('

        for col in tbls[t]:
            sql += col[0] + ' ' + col[1] + ', '  # Append a pair <col_name> <data-type>

        sql = sql[:-2] + ')'    # Replace the last ', ' with ')'
        cursor.execute(sql)     # Create table

    if return_cursor:
        return cursor

    conn.close()
    return None

