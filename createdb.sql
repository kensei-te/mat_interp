CREATE DATABASE IF NOT EXISTS Mat_interp;
CREATE USER IF NOT EXISTS mat_user_1@localhost IDENTIFIED BY 'mat_user_1_P';
GRANT ALL PRIVILEGES ON Mat_interp.* TO mat_user_1@localhost;
FLUSH PRIVILEGES;