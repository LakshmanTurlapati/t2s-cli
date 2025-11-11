#!/usr/bin/env python3
"""
Migrate Chinook SQLite database to MongoDB.

This script reads all tables from the Chinook SQLite database and migrates them
to MongoDB collections in a database called 'chinook_mongo'.
"""

import sqlite3
import sys
from pathlib import Path
from pymongo import MongoClient
from datetime import datetime


def connect_sqlite(db_path):
    """Connect to SQLite database."""
    print(f"Connecting to SQLite database: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def connect_mongodb(host="localhost", port=27017):
    """Connect to MongoDB."""
    print(f"Connecting to MongoDB at {host}:{port}")
    client = MongoClient(host, port)
    return client


def get_table_names(sqlite_conn):
    """Get all table names from SQLite database."""
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [row[0] for row in cursor.fetchall()]


def get_table_data(sqlite_conn, table_name):
    """Get all data from a table."""
    cursor = sqlite_conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    return columns, rows


def row_to_document(row, columns):
    """Convert SQLite row to MongoDB document."""
    doc = {}
    for i, col in enumerate(columns):
        value = row[i]
        # Convert None to null, keep other values as-is
        doc[col] = value
    return doc


def migrate_table(sqlite_conn, mongo_db, table_name):
    """Migrate a single table to MongoDB collection."""
    print(f"\nMigrating table: {table_name}")

    # Get table data
    columns, rows = get_table_data(sqlite_conn, table_name)
    print(f"  Columns: {', '.join(columns)}")
    print(f"  Rows: {len(rows)}")

    if len(rows) == 0:
        print(f"  Skipping empty table")
        return

    # Convert rows to documents
    documents = []
    for row in rows:
        doc = row_to_document(row, columns)
        documents.append(doc)

    # Insert into MongoDB
    collection = mongo_db[table_name]

    # Drop collection if it exists
    collection.drop()

    # Insert documents
    result = collection.insert_many(documents)
    print(f"  Inserted {len(result.inserted_ids)} documents into collection '{table_name}'")

    return len(result.inserted_ids)


def create_indexes(mongo_db):
    """Create indexes on MongoDB collections to match SQLite foreign keys."""
    print("\nCreating indexes...")

    # Album indexes
    mongo_db.Album.create_index("AlbumId", unique=True)
    mongo_db.Album.create_index("ArtistId")

    # Artist indexes
    mongo_db.Artist.create_index("ArtistId", unique=True)

    # Customer indexes
    mongo_db.Customer.create_index("CustomerId", unique=True)
    mongo_db.Customer.create_index("SupportRepId")

    # Employee indexes
    mongo_db.Employee.create_index("EmployeeId", unique=True)
    mongo_db.Employee.create_index("ReportsTo")

    # Genre indexes
    mongo_db.Genre.create_index("GenreId", unique=True)

    # Invoice indexes
    mongo_db.Invoice.create_index("InvoiceId", unique=True)
    mongo_db.Invoice.create_index("CustomerId")

    # InvoiceLine indexes
    mongo_db.InvoiceLine.create_index("InvoiceLineId", unique=True)
    mongo_db.InvoiceLine.create_index("InvoiceId")
    mongo_db.InvoiceLine.create_index("TrackId")

    # MediaType indexes
    mongo_db.MediaType.create_index("MediaTypeId", unique=True)

    # Playlist indexes
    mongo_db.Playlist.create_index("PlaylistId", unique=True)

    # PlaylistTrack indexes (composite primary key)
    mongo_db.PlaylistTrack.create_index([("PlaylistId", 1), ("TrackId", 1)], unique=True)
    mongo_db.PlaylistTrack.create_index("TrackId")

    # Track indexes
    mongo_db.Track.create_index("TrackId", unique=True)
    mongo_db.Track.create_index("AlbumId")
    mongo_db.Track.create_index("MediaTypeId")
    mongo_db.Track.create_index("GenreId")

    print("  Indexes created successfully")


def verify_migration(sqlite_conn, mongo_db):
    """Verify that the migration was successful by comparing counts."""
    print("\nVerifying migration...")

    table_names = get_table_names(sqlite_conn)
    all_match = True

    for table_name in table_names:
        # Get SQLite count
        cursor = sqlite_conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        sqlite_count = cursor.fetchone()[0]

        # Get MongoDB count
        mongo_count = mongo_db[table_name].count_documents({})

        match = "✓" if sqlite_count == mongo_count else "✗"
        print(f"  {match} {table_name}: SQLite={sqlite_count}, MongoDB={mongo_count}")

        if sqlite_count != mongo_count:
            all_match = False

    return all_match


def main():
    """Main migration function."""
    print("=" * 60)
    print("Chinook SQLite to MongoDB Migration")
    print("=" * 60)

    # Paths and connection details
    sqlite_db_path = Path(__file__).parent.parent / "t2s" / "chinook.db"
    mongo_host = "localhost"
    mongo_port = 27017
    mongo_db_name = "chinook_mongo"

    if not sqlite_db_path.exists():
        print(f"Error: SQLite database not found at {sqlite_db_path}")
        sys.exit(1)

    try:
        # Connect to databases
        sqlite_conn = connect_sqlite(sqlite_db_path)
        mongo_client = connect_mongodb(mongo_host, mongo_port)
        mongo_db = mongo_client[mongo_db_name]

        # Get all table names
        table_names = get_table_names(sqlite_conn)
        print(f"\nFound {len(table_names)} tables: {', '.join(table_names)}")

        # Migrate each table
        total_docs = 0
        for table_name in table_names:
            docs_inserted = migrate_table(sqlite_conn, mongo_db, table_name)
            if docs_inserted:
                total_docs += docs_inserted

        # Create indexes
        create_indexes(mongo_db)

        # Verify migration
        success = verify_migration(sqlite_conn, mongo_db)

        print("\n" + "=" * 60)
        if success:
            print(f"Migration completed successfully!")
            print(f"Total documents inserted: {total_docs}")
            print(f"Database: {mongo_db_name}")
            print(f"Connection: mongodb://{mongo_host}:{mongo_port}")
        else:
            print("Migration completed with warnings - some counts don't match")
        print("=" * 60)

        # Close connections
        sqlite_conn.close()
        mongo_client.close()

        return 0 if success else 1

    except Exception as e:
        print(f"\nError during migration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
