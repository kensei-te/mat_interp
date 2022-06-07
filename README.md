# Mat interp

[Overview](#overview)  
[Operating System](#operating-system)  
[How to get ready (install)](#how-to-get-ready-install)  
[Usage](#usage)  

## Overview
This code is to perform neural networks learning of preliminary or repository experimental data, to simulate future experiments.  
For details, refer to:

![gif](overview.gif)
<br />
<br />

## Operating System
It supports Linux(Ubuntu and CentOS) and MacOS, but does _not_ cover Windows, to avoid complexity arising from OS-dependency.  Linux can be prepared in Windows using OS virtualization softwares.
<br />
<br />

## How to get ready (install)

### 1. Scripts and Environment Setup
We recommend using conda for installing.
1. Download this repository (git clone or simply 
download)
2. Create the environment using:
	```bash
	conda env create -f=environment.yml
	```

3. Activate the environment (name of environment is 'mat_interp' by default, it can be changed by modifying environment.yml):
	```bash
	conda activate mat_interp
	```
4. After this is done, follow the below procedures for setting up MySQL server (local database used to take care data of optimization process of neural networks architeture in parallel.)




### 2. MySQL Installation
The procedure slightly differs between Ubuntu, CentOS, and MacOS.


<details><summary>Ubuntu 20.04</summary><div>

#### Installation
1. update apt just in case  
	```bash
	sudo apt update
	```
2. check available package
	```bash
	apt-cache policy mysql-server
	```
	...suppose "8.0.22-0ubuntu0.20.04.2" is shown as candidate,
3. simulate installation
	```bash
	apt-get install -s mysql-client=8.0.22-0ubuntu0.20.04.2
	```
	```bash
	apt-get install -s mysql-community-server=8.0.22-0ubuntu0.20.04.2
	```
	```bash
	apt-get install -s mysql-server=8.0.22-0ubuntu0.20.04.2
	```
	if there is no error, let's install them
4. install
	```bash
	apt-get install mysql-client=8.0.22-0ubuntu0.20.04.2
	```
	```bash
	apt-get install mysql-community-server=8.0.22-0ubuntu0.20.04.2
	```
	```bash
	apt-get install mysql-server=8.0.22-0ubuntu0.20.04.2
	```
#### Setup
1. set password for root
	```bash
	sudo mysql-secure_installation
	```
	set your password
2. keep answering yes, until the script "mysql-secure_installation" ends
3. login to MySQL, using the password set above  
	```bash
	mysql -u root -p
	```
4. create a database, called “Mat_interp”
	```bash
	mysql> CREATE DATABASE IF NOT EXISTS Mat_interp;
	```
5. create user, username: mat_user_1, password:mat_user_1_P  
	```bash
	mysql> CREATE USER IF NOT EXISTS mat_user_1@localhost IDENTIFIED BY ‘mat_user_1_P’;
	```
6. allow user to modify only database named “Mat_interp”
	```bash
	mysql> GRANT ALL PRIVILEGES ON Mat_interp.* TO mat_user_1@localhost;
	```
7. release memory
	```bash
	mysql> FLUSH PRIVILEGES;
	```
8. exit mysql_console
	```bash
	mysql> exit;
	```
9. make a configuration file “.my.cnf” for mysql, where username etc is written at home folder.  Here we write password for user1 who is allowed to modify only Mat_interp database.  From now on, config.ini will be referenced from executing app.py, and you do not need to enter password each time.  You can modify the setting to make things safer.

	- at your home directory:  
		```bash
		vi .my.cnf
		```
	[client]  
	user = mat_user_1  
	password = mat_user_1_P  

	- at Mat_interp folder:  
	modify config.ini if you change username and password

10. confine the accessibility of those files to the current user only
	```bash
	chmod 600 .my.cnf
	chmod 600 Mat_conf
	```

11. check if mysql is booted  
	```bash
	systemctl status mysqld
	```

    - in case it is not started:  
		```bash
		sudo systemctl start mysqld
		```  
    - in case you want to stop mysql:  
		```bash
		sudo systemctl stop mysql  
		```
</div></details>


<details><summary>CentOS7</summary><div> 

#### Installation
1. delete MariaDB, which is installed by default but may compete with MySQL  
	- make sure what MariaDB packages you have  
		```bash
		rpm -qa | grep aria
		```
	- remove MariDB-related things  
		```bash
		% yum remove mariadb-libs
		```
2.  enable access to repository for MySQL
	- download yum-repository, go to 
		http://dev.mysql.com/downloads/repo/yum/
		and choose RPM for “Red Hat Enterprise Linux 7”  
3. install the downloaded RPM  
	```bash
	yum localinstall   mysql80-community-release-el7-3.noarch.rpm
	```
4. install MySQL8.0  
	```bash
	yum install —enablerepo=mysql80-community mysql-community-server
	```

#### Setup
1. make sure the initial password for root  
	```bash
	cat /var/log/mysqld.log | grep password
	```
2. login to MySQL, using the initial password above  
	```bash
	mysql -u root -p
	```
3. change password in mysql_console (after logging in)
	```bash
	mysql> set password for root@localhost=‘new_password’;
	```
4. create a database, called “Mat_interp”
	```bash
	mysql> CREATE DATABASE IF NOT EXISTS Mat_interp;
	```
5. create user, username: mat_user_1, password:mat_user_1_P  
	```bash
	mysql> CREATE USER IF NOT EXISTS mat_user_1@localhost IDENTIFIED BY ‘mat_user_1_P’;
	```
6. allow user to modify only database named “Mat_interp”
	```bash
	mysql> GRANT ALL PRIVILEGES ON Mat_interp.* TO mat_user_1@localhost;
	```
7. release memory
	```bash
	mysql> FLUSH PRIVILEGES;
	```
8. exit mysql_console
	```bash
	mysql> exit;
	```
9. make a configuration file “.my.cnf” for mysql, where username etc is written at home folder.  Here we write password for user1 who is allowed to modify only Mat_interp database.  From now on, config.ini will be referenced from executing app.py, and you do not need to enter password each time.  You can modify the setting to make things safer.

	- at your home directory:  
		```bash
		vi .my.cnf
		```
		[client]  
		user = mat_user_1  
		password = mat_user_1_P

	- at Mat_interp folder:  
		modify config.ini if you change username and password

10. confine the accessibility of those files to the current user only
	```bash
	chmod 600 .my.cnf
	chmod 600 Mat_conf
	```

11. check if mysql is booted  
	```bash
	systemctl status mysqld
	```

    - in case it is not started:  
		```bash
		sudo systemctl start mysqld
		```
    - in case you want to stop mysql:  
		```bash
		sudo systemctl stop mysql  
		```
</div></details>



<details><summary>MacOS</summary><div>

#### Installation
1. (In case it is not installed) install Homebrew, which helps installation of MySQL
	```bash
	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	```
2. Install MySQL using Homebrew
	```bash
	brew install mysql@8.0
	```
3. set path for mysql
	```bash
	echo 'export PATH="/usr/local/opt/mysql@8.0/bin:$pATH"' >> ~/.bashrc
	```
	in case you are using other shell, modify ".bashrc" to corresponding one
4. set path for mysql
	```bash
	source ~/.bashrc
	```
	in case you are using other shell, modify ".bashrc" to corresponding one
#### Setup
1. set password for root
	% sudo mysql-secure_installation
	set your password
2. keep answering yes, until the script "mysql-secure_installation" ends
3. change password in mysql_console (after logging in)
	```bash
	mysql> set password for root@localhost=‘new_password’;
	```
4. create a database, called “Mat_interp”
	```bash
	mysql> CREATE DATABASE IF NOT EXISTS Mat_interp;
	```
5. create user, username: mat_user_1, password:mat_user_1_P  
	```bash
	mysql> CREATE USER IF NOT EXISTS mat_user_1@localhost IDENTIFIED BY ‘mat_user_1_P’;
	```
6. allow user to modify only database named “Mat_interp”
	```bash
	mysql> GRANT ALL PRIVILEGES ON Mat_interp.* TO mat_user_1@localhost;
	```
7. release memory
	```bash
	mysql> FLUSH PRIVILEGES;
	```
8. exit mysql_console
	```bash
	mysql> exit;
	```
9. make a configuration file “.my.cnf” for mysql, where username etc is written at home folder.  Here we write password for user1 who is allowed to modify only Mat_interp database.  From now on, config.ini will be referenced from executing app.py, and you do not need to enter password each time.  You can modify the setting to make things safer.

	- at your home directory:  
		```bash
		vi .my.cnf
		```
		[client]  
		user = mat_user_1  
		password = mat_user_1_P

	- at Mat_interp folder:  
		modify config.ini if you change username and password

10. confine the accessibility of those files to the current user only
	```bash
	chmod 600 .my.cnf
	chmod 600 Mat_conf
	```

11. check if mysql is booted  
	```bash
	systemctl status mysqld
	```

    - in case it is not started:  
		```bash
		sudo systemctl start mysqld
		```
    - in case you want to stop mysql:  
		```bash
		sudo systemctl stop mysql  
		```

</div></details>




<br />


## Usage
### Launch app
  1. in case you have environment, activate the environment
		```bash
		conda activate 'your_environment'
		```
  2. if not booted yet, boot mysql (database to store learning result):
		```bash
		sudo systemctl start mysqld
		```
  3. change currnt directory to Mat_interp folder:
		```bash
		cd (your_directory)/Mat_interp
		```
  4. start streamlit app (GUI):
		```bash
		streamlit run app.py
		```
  5. access to streamlit GUI from your web_browser:
	http://127.0.0.1:8501/



### Use
  1. prepare csv file of your experimental data
  1. enter working folder name
  1. upload the csv file
  1. here we let machine learn Y(X1, X2)  where Y is target value, X1 and X2 are features (experimental scanning axes).  choose X1, X2, Y column names
  1. in case you want to check data, check visualize data
  2. choose number of total trials, number of parallel workers, and learning algorithm.
  2. click save NN_setting, to save settings above.
  2. click execute learning
  2. go to “2.5”
  2. click “check/update results”
  2.  When learning is finished, choose whether you continue to search or use the best model so far.
  3.  Specify X1 and X2 range to predict corresponding Y values.
  3.  click “simulate”
  3.  check “visualize”
  3.  download csv of simulated data


  




