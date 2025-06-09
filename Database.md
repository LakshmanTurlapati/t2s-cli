# Chinook Database Overview

## About the Database
The **Chinook Database** is a sample database that represents a digital media store. It's based on the iTunes Store and contains information about music tracks, albums, artists, customers, employees, and sales transactions.

## Database Schema

### Core Tables

**🎵 Music Catalog:**
- **artists** - Musical artists (275 records)
  - `ArtistId` (PK), `Name`
- **albums** - Music albums (347 records) 
  - `AlbumId` (PK), `Title`, `ArtistId` (FK to artists)
- **tracks** - Individual songs (3,503 records)
  - `TrackId` (PK), `Name`, `AlbumId` (FK), `MediaTypeId` (FK), `GenreId` (FK), `Composer`, `Milliseconds`, `Bytes`, `UnitPrice`
- **genres** - Music genres (25 records)
  - `GenreId` (PK), `Name`
- **media_types** - File formats (5 records)
  - `MediaTypeId` (PK), `Name`

**👥 Customer Management:**
- **customers** - Store customers (59 records)
  - `CustomerId` (PK), `FirstName`, `LastName`, `Company`, `Address`, `City`, `State`, `Country`, `PostalCode`, `Phone`, `Fax`, `Email`, `SupportRepId` (FK to employees)
- **employees** - Company employees (8 records)
  - `EmployeeId` (PK), `LastName`, `FirstName`, `Title`, `ReportsTo` (FK to employees), `BirthDate`, `HireDate`, `Address`, `City`, `State`, `Country`, `PostalCode`, `Phone`, `Fax`, `Email`

**💰 Sales & Transactions:**
- **invoices** - Customer purchase records (412 records)
  - `InvoiceId` (PK), `CustomerId` (FK), `InvoiceDate`, `BillingAddress`, `BillingCity`, `BillingState`, `BillingCountry`, `BillingPostalCode`, `Total`
- **invoice_items** - Individual items on invoices (2,240 records)
  - `InvoiceLineId` (PK), `InvoiceId` (FK), `TrackId` (FK), `UnitPrice`, `Quantity`

**🎵 Playlists:**
- **playlists** - Music playlists (18 records)
  - `PlaylistId` (PK), `Name`
- **playlist_track** - Tracks in playlists (8,715 records)
  - `PlaylistId` (FK), `TrackId` (FK)

## Key Relationships
- Artists → Albums (one-to-many)
- Albums → Tracks (one-to-many)
- Genres → Tracks (one-to-many)
- MediaTypes → Tracks (one-to-many)
- Customers → Invoices (one-to-many)
- Employees → Customers (one-to-many support relationship)
- Invoices → InvoiceItems (one-to-many)
- Tracks → InvoiceItems (one-to-many)
- Playlists ↔ Tracks (many-to-many via playlist_track)

---

## 10 Sample Queries to Try

### 1. Basic Counts
```
How many artists are in the database?
```

### 2. Genre Analysis  
```
What are all the music genres available?
```

### 3. Top Artists
```
Which artist has the most albums?
```

### 4. Sales Analysis
```
What is the total revenue from all sales?
```

### 5. Customer Geography
```
How many customers are from each country?
```

### 6. Popular Tracks
```
Which tracks have been purchased the most times?
```

### 7. Employee Performance
```
Which employee has the most customers assigned to them?
```

### 8. Album Duration
```
What is the longest album by total track time?
```

### 9. Price Analysis
```
What are the most expensive tracks in the store?
```

### 10. Complex Join Query
```
Show me the top 5 customers by total amount spent along with their support representative
```

---

## Usage Tips
- The database contains real-world-like data with proper relationships
- All monetary values are in USD
- Track lengths are in milliseconds
- File sizes are in bytes
- Great for testing SQL JOINs, aggregations, and complex queries
- Perfect for learning database concepts and SQL optimization

---

## T2S Testing
This database is ideal for testing the T2S (Text-to-SQL) system because it:
- Has clear, intuitive table names and relationships
- Contains various data types (text, numbers, dates)
- Supports both simple and complex queries
- Includes realistic business scenarios
- Has enough data volume to make queries meaningful 