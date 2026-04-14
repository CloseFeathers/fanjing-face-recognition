-- Chinook PostgreSQL Seed — mirrors SQLite migration_database.sqlite schema
-- All 11 tables with foreign keys, minimal representative data.
-- Uses unquoted lowercase identifiers (PostgreSQL convention).

CREATE TABLE IF NOT EXISTS artist (
    artistid INTEGER PRIMARY KEY,
    name VARCHAR(120)
);

CREATE TABLE IF NOT EXISTS genre (
    genreid INTEGER PRIMARY KEY,
    name VARCHAR(120)
);

CREATE TABLE IF NOT EXISTS mediatype (
    mediatypeid INTEGER PRIMARY KEY,
    name VARCHAR(120)
);

CREATE TABLE IF NOT EXISTS playlist (
    playlistid INTEGER PRIMARY KEY,
    name VARCHAR(120)
);

CREATE TABLE IF NOT EXISTS employee (
    employeeid INTEGER PRIMARY KEY,
    lastname VARCHAR(20) NOT NULL,
    firstname VARCHAR(20) NOT NULL,
    title VARCHAR(30),
    reportsto INTEGER REFERENCES employee(employeeid),
    birthdate TIMESTAMP,
    hiredate TIMESTAMP,
    address VARCHAR(70),
    city VARCHAR(40),
    state VARCHAR(40),
    country VARCHAR(40),
    postalcode VARCHAR(10),
    phone VARCHAR(24),
    fax VARCHAR(24),
    email VARCHAR(60)
);

CREATE TABLE IF NOT EXISTS album (
    albumid INTEGER PRIMARY KEY,
    title VARCHAR(160) NOT NULL,
    artistid INTEGER NOT NULL REFERENCES artist(artistid)
);

CREATE TABLE IF NOT EXISTS customer (
    customerid INTEGER PRIMARY KEY,
    firstname VARCHAR(40) NOT NULL,
    lastname VARCHAR(20) NOT NULL,
    company VARCHAR(80),
    address VARCHAR(70),
    city VARCHAR(40),
    state VARCHAR(40),
    country VARCHAR(40),
    postalcode VARCHAR(10),
    phone VARCHAR(24),
    fax VARCHAR(24),
    email VARCHAR(60) NOT NULL,
    supportrepid INTEGER REFERENCES employee(employeeid)
);

CREATE TABLE IF NOT EXISTS track (
    trackid INTEGER PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    albumid INTEGER REFERENCES album(albumid),
    mediatypeid INTEGER NOT NULL REFERENCES mediatype(mediatypeid),
    genreid INTEGER REFERENCES genre(genreid),
    composer VARCHAR(220),
    milliseconds INTEGER NOT NULL,
    bytes INTEGER,
    unitprice NUMERIC(10,2) NOT NULL
);

CREATE TABLE IF NOT EXISTS invoice (
    invoiceid INTEGER PRIMARY KEY,
    customerid INTEGER NOT NULL REFERENCES customer(customerid),
    invoicedate TIMESTAMP NOT NULL,
    billingaddress VARCHAR(70),
    billingcity VARCHAR(40),
    billingstate VARCHAR(40),
    billingcountry VARCHAR(40),
    billingpostalcode VARCHAR(10),
    total NUMERIC(10,2) NOT NULL
);

CREATE TABLE IF NOT EXISTS invoiceline (
    invoicelineid INTEGER PRIMARY KEY,
    invoiceid INTEGER NOT NULL REFERENCES invoice(invoiceid),
    trackid INTEGER NOT NULL REFERENCES track(trackid),
    unitprice NUMERIC(10,2) NOT NULL,
    quantity INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS playlisttrack (
    playlistid INTEGER NOT NULL REFERENCES playlist(playlistid),
    trackid INTEGER NOT NULL REFERENCES track(trackid),
    PRIMARY KEY (playlistid, trackid)
);

-- =============================================================================
-- Sample data (mirrors migration_database.sqlite)
-- =============================================================================

INSERT INTO artist (artistid, name) VALUES
    (1, 'AC/DC'), (2, 'Accept'), (3, 'Aerosmith');

INSERT INTO genre (genreid, name) VALUES (1, 'Rock');

INSERT INTO mediatype (mediatypeid, name) VALUES
    (1, 'MPEG audio file'), (2, 'Protected AAC audio file');

INSERT INTO playlist (playlistid, name) VALUES
    (1, 'Music'), (2, 'Movies'), (3, 'TV Shows');

INSERT INTO employee (employeeid, lastname, firstname, title, reportsto) VALUES
    (1, 'Adams', 'Andrew', 'General Manager', NULL),
    (2, 'Edwards', 'Nancy', 'Sales Manager', 1),
    (3, 'Peacock', 'Jane', 'Sales Support Agent', 2),
    (4, 'Park', 'Margaret', 'Sales Support Agent', 2),
    (5, 'Johnson', 'Steve', 'Sales Support Agent', 2),
    (6, 'Mitchell', 'Michael', 'IT Manager', 1),
    (7, 'King', 'Robert', 'IT Staff', 6),
    (8, 'Callahan', 'Laura', 'IT Staff', 6);

INSERT INTO album (albumid, title, artistid) VALUES
    (1, 'For Those About To Rock We Salute You', 1),
    (2, 'Balls to the Wall', 2),
    (3, 'Restless and Wild', 2),
    (4, 'Let There Be Rock', 1),
    (5, 'Big Ones', 3);

INSERT INTO customer (customerid, firstname, lastname, company, email, supportrepid) VALUES
    (1, 'Luís', 'Gonçalves', 'Embraer', 'luisg@embraer.com.br', 3),
    (2, 'Leonie', 'Köhler', NULL, 'leonekohler@surfeu.de', 5),
    (3, 'François', 'Tremblay', NULL, 'ftremblay@gmail.com', 3);

INSERT INTO track (trackid, name, albumid, mediatypeid, genreid, milliseconds, unitprice) VALUES
    (1, 'For Those About To Rock', 1, 1, 1, 343719, 0.99),
    (2, 'Balls to the Wall', 2, 2, 1, 342562, 0.99),
    (3, 'Fast As a Shark', 3, 2, 1, 230619, 0.99),
    (4, 'Restless and Wild', 3, 2, 1, 252051, 0.99),
    (5, 'Princess of the Dawn', 3, 2, 1, 375418, 0.99);

INSERT INTO invoice (invoiceid, customerid, invoicedate, total) VALUES
    (1, 2, '2009-01-01', 1.98),
    (2, 1, '2009-01-02', 3.96),
    (3, 3, '2009-01-03', 5.94);

INSERT INTO invoiceline (invoicelineid, invoiceid, trackid, unitprice, quantity) VALUES
    (1, 1, 2, 0.99, 1),
    (2, 1, 4, 0.99, 1),
    (3, 2, 1, 0.99, 2),
    (4, 3, 3, 0.99, 1),
    (5, 3, 5, 0.99, 1);

INSERT INTO playlisttrack (playlistid, trackid) VALUES
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 1), (3, 3);
