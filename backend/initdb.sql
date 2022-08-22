CREATE TABLE IF NOT EXISTS lines_usage (
    id INTEGER PRIMARY KEY,
    date DATE NOT NULL,
    time_slot INTEGER NOT NULL DEFAULT 0,
    reservation text NOT NULL
);

CREATE TABLE IF NOT EXISTS occupancy (
    id INTEGER PRIMARY KEY,
    percent INTEGER NOT NULL DEFAULT 0,
    pool INTEGER NOT NULL DEFAULT 0,
    park INTEGER NOT NULL DEFAULT 0,
    lines_reserved INTEGER NOT NULL DEFAULT 0,
    time timestamp NOT NULL,
    day_of_week INTEGER NOT NULL
);
