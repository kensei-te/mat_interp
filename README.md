# NN interp

## Overview
This code is to perform neural networks learning of preliminary or repository experimental data, to simulate future experiments. 
For details, refer to:

![gif](overview.gif)

## How to get ready (install)

### Scripts and Environment Setup
We recommend using conda for installing.
1. Download this repository (git clone or simply 
download)
2. Create the enviroment using:
```bash
conda create env -f environment.yml
```

3. Activate the environment:
```bash
conda activate nninterp
```
4. After this is done, follow the below procedures for setting up MySQL server (Used to run optimization of neural networks architeture in parallel.)






### MySQL Installation for Optimization (CentOS7)
#### Installation
1. delete MariaDB, which is installed by default but may compete with MySQL  
	- make sure what MariaDB packages you have  
		% rpm -qa | grep aria
	- remove MariDB-related things  
		% yum remove mariadb-libs
2.  enable access to repository for MySQL
	- download yum-repository, go to 
		http://dev.mysql.com/downloads/repo/yum/
		and choose RPM for “Red Hat Enterprise Linux 7”  
3. install the downloaded RPM  
	% yum localinstall   mysql80-community-release-el7-3.noarch.rpm
4. install MySQL8.0  
	% yum install —enablerepo=mysql80-community mysql-community-server

#### Setup
1. make sure the initial password for root  
	% cat /var/log/mysqld.log | grep password  
2. login to MySQL, using the initial password above  
	% mysql -u root -p  
3. change password in mysql_console (after logging in)
	mysql> set password for root@localhost=‘new_password’;
4. create a database, called “NN_interp”
	mysql> CREATE DATABASE IF NOT EXISTS NN_interp;
5. create user, username: nn_user_1, password:nn_user_1_P  
	mysql> CREATE USER IF NOT EXISTS nn_user_1@localhost IDENTIFIED BY ‘nn_user_1_P’;
6. allow user to modify only database named “NN_interp”
	mysql> GRANT ALL PRIVILEGES ON NN_interp.* TO nn_user_1@localhost;
7. release memory
	mysql> FLUSH PRIVILEGES;
8. exit mysql_console
	mysql> exit;
9. make a configuration file “.my.cnf” for mysql, where username etc is written  
	at home folder.  Here we write password for user1 who is allowed to modify only NN_interp database.  From now on, config.ini will be referenced from executing python file of NN_interp, and you do not need to enter password each time.  You can modify the setting to make things safer.

	- at your home directory:  
	% vi .my.cnf
	[client]
	user = nn_user_1
	password = nn_user_1_P

	- at NN_interp folder:  
	modify config.ini if you change username and password

10. confine the accessibility of those files to the current user only
	% chmod 600 .my.cnf
	% chmod 600 NN_conf

11. check if mysql is booted  
	% systemctl status mysqld

    - in case it is not started:  
	% sudo systemctl start mysqld  
    - in case you want to stop mysql:  
	% sudo systemctl stop mysql  


## Usage
### Launch app
  1. in case you have environment, activate the environment
  2. if not booted yet, boot mysql (database to store learning result):
	% sudo systemctl start mysqld
  3. change currnt directory to NN_interp folder:
	% cd (your_directory)/NN_interp
  4. start streamlit app (GUI):
	% streamlit run NN_interp.py
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


  




