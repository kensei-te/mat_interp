# Mat interp

[Overview](#overview)  
[Operating System](#operating-system)  
[How to get ready (install)](#how-to-get-ready-install)  
[Usage](#usage)  
[Uninstall](#uninstall)

## Overview
This code is to perform neural networks learning of preliminary or repository experimental data, to simulate future experiments.  
For details, refer to:

![gif](overview.gif)
<br />
<br />

## Operating System
It works in Linux and MacOS(Intel), but does _not_ cover Windows, to avoid complexity arising from OS-dependency.  Linux can be prepared in Windows using OS virtualization softwares.  MacOS(M1) might work but not verified yet (See requirement.txt and https://github.com/apple/tensorflow_macos)

Verified to work: Ubuntu20.04, Ubuntu18.04, CentOS7, MacOS11(Intel), MacOS12(Intel)
<br />
<br />

## How to get ready (install)

### 1. Scripts and Environment Setup
We recommend using conda for installing.
1. Download this repository (git clone or simply 
download), and move to the directory (default name: 'mat_interp')
	```bash
	cd (your_directory)/mat_interp
	```
2. Create the environment using env_***.yml  
	For Ubuntu:
	```bash
	conda env create -f=env_Ubuntu.yml
	```
	For CentOS7:  
	```bash
	conda env create -f=env_CentOS.yml
	```
	For MacOS(Intel):  
	```bash
	conda env create -f=env_mac.yml
	```


3. Activate the environment (name of environment is 'mat_interp' by default, it can be changed by modifying environment.yml):
	```bash
	conda activate mat_interp
	```
4. After this is done, follow the below procedures for setting up MySQL server (local database used to take care data of optimization process of neural networks architeture in parallel.)




### 2. MySQL Installation
The procedure slightly differs between Ubuntu, CentOS, and MacOS.


<details><summary>Ubuntu</summary><div>

#### Installation
1. update apt just in case  
	```bash
	sudo apt update
	```
2. check available package
	```bash
	apt-cache policy mysql-server
	```
	v.8.0.x or v.5.7.x are recommended.  
	...suppose "8.0.22-0ubuntu0.20.04.2" is shown as candidate,
3. simulate installation
	```bash
	apt-get install -s mysql-server=8.0.22-0ubuntu0.20.04.2
	```
	if there is no error, let's install them
4. install
	```bash
	sudo apt-get install mysql-server=8.0.22-0ubuntu0.20.04.2
	```
	mysql-server, mysql-client and other required packages will be installed.
#### Setup
1. set password for root
	```bash
	sudo mysql_secure_installation
	```
	set your password
2. keep answering yes, until the script "mysql_secure_installation" ends
3. login to MySQL  
	```bash
	sudo mysql
	```
4. run the SQL script file then exit:
	```bash
	mysql> source createdb.sql
	mysql> exit;
	```
	The script will make a database named "Mat_interp", and make a user "mat_user_1" with password, and give a privilege to this user to modify the database Mat_interp. You can modify them by editing createdb.sql.
5. edit a configuration file ".my.cnf" for mysql.  Here we write password for user1 who is allowed to modify only Mat_interp database.  

	```bash  
	vi ~/.my.cnf  
	```  
	you can use other editors as well. add(write) the following to .my.cnf and save  

	[client]  
	user = mat_user_1  
	password = mat_user_1_P  
	
	Then, confine the accessibility of the file to the current user only  
	```bash  
	chmod 600 ~/.my.cnf  
	```  

	modify config.ini in mat_interp folder if you change username and password
	
	then, confine the accessibility of the file to the current user only:
	```bash
	chmod 600 config.ini
	```
	From now on, "config.ini" will be referenced from executing "app.py", and you do not need to enter password each time.  You can modify the setting to make things safer.

6. check if mysql is booted  
	```bash
	systemctl status mysql
	```
    - in case it is not started:  
		```bash
		sudo systemctl start mysql
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
4. run the SQL script file then exit:
	```bash
	mysql> source createdb.sql
	mysql> exit;
	```
	The script will make a database named "Mat_interp", and make a user "mat_user_1" with password, and give a privilege to this user to modify the database Mat_interp. You can modify them by editing createdb.sql.
5. edit a configuration file ".my.cnf" for mysql.  Here we write password for user1 who is allowed to modify only Mat_interp database.

	```bash  
	vi ~/.my.cnf  
	```  
	you can use other editors as well. add(write) the following to .my.cnf and save  

	[client]  
	user = mat_user_1  
	password = mat_user_1_P  
	
	Then, confine the accessibility of the file to the current user only  
	```bash  
	chmod 600 ~/.my.cnf  
	```  

	modify config.ini in mat_interp folder if you change username and password
	
	then, confine the accessibility of the file to the current user only:
	```bash
	chmod 600 config.ini
	```
	From now on, "config.ini" will be referenced from executing "app.py", and you do not need to enter password each time.  You can modify the setting to make things safer.  

6. check if mysql is booted  
	```bash
	systemctl status mysqld
	```

    - in case it is not started:  
		```bash
		sudo systemctl start mysqld
		```
    - in case you want to stop mysql:  
		```bash
		sudo systemctl stop mysqld  
		```
</div></details>



<details><summary>MacOS(Intel)</summary><div>

#### Installation
1. (In case it is not installed) install Homebrew, which helps installation of MySQL
	```bash
	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	```
2. Install MySQL using Homebrew
	```bash
	brew install mysql@8.0
	```

<!--
3. set path for mysql
	```bash
	echo 'export PATH="/usr/local/opt/mysql@8.0/bin:$PATH"' >> ~/.zshrc  
	source ~/.zshrc
	```
	in case you are using other shell, modify ".zshrc" to corresponding one
-->

#### Setup
1. launch mysql and set password for root
	```bash  
	mysql.server start
	mysql_secure_installation
	```
	set your password
2. keep answering yes, until the script "mysql_secure_installation" ends
3. login to mysql, using the password set above 
	```bash
	mysql -u root -p
	```
4. run the SQL script file then exit:
	```bash
	mysql> source createdb.sql
	mysql> exit;
	```
	The script will make a database named "Mat_interp", and make a user "mat_user_1" with password, and give a privilege to this user to modify the database Mat_interp. You can modify them by editing createdb.sql.
5. edit a configuration file ".my.cnf" for mysql.  Here we write password for user1 who is allowed to modify only Mat_interp database.  

	```bash  
	vi ~/.my.cnf  
	```  
	you can use other editors as well. add(write) the following to .my.cnf and save  

	[client]  
	user = mat_user_1  
	password = mat_user_1_P  
	
	Then, confine the accessibility of the file to the current user only  
	```bash  
	chmod 600 ~/.my.cnf  
	```  

	modify config.ini in mat_interp folder if you change username and password
	
	then, confine the accessibility of the file to the current user only:
	```bash
	chmod 600 config.ini
	```
	From now on, "config.ini" will be referenced from executing "app.py", and you do not need to enter password each time.  You can modify the setting to make things safer.

6. check if mysql server is booted  
	```bash
	mysql.server status
	```

    - in case it is not started:  
		```bash
		mysql.server start
		```
    - in case you want to stop mysql server:  
		```bash
		mysql.server stop  
		```
</div></details>




<br />


## Usage
### Launch app
  1. in case you have environment, activate the environment (default is 'mat_interp')
		```bash
		conda activate mat_interp
		```
  2. if not booted yet, boot mysql (database to store learning result):
	
		- Ubuntu:
			```bash
			sudo systemctl start mysqld
			```  

		- CentOS7  
			```bash
			sudo systemctl start mysql
			```  

		- MacOS(Intel)
			```bash
			mysql.server start
			```		
  3. change current directory to mat_interp folder:
		```bash
		cd (your_directory)/mat_interp
		```
  4. start streamlit app (GUI):
		```bash
		streamlit run app.py
		```
		it may take time to launch in the very first time  
		Ubuntu users may need to downgrade protobuf
		```bash
		pip install protobuf==3.19.0
		```
  5. access to streamlit GUI from your web_browser:
	http://127.0.0.1:8501/

  6. (after use)  
  	to stop streamlit, close web_browser, then type "ctrl" + "." in terminal.  
	to stop mysql, for Ubuntu:
		```bash
		sudo systemctl stop mysql
		```
		for CentOS
		```bash
		sudo systemctl stop mysqld
		```  
		for MacOS(Intel)
		```bash
		mysql.server stop
		```
	to deactivate virtual environment (named "mat_interp"),
		```bash
		conda deactivate mat_interp
		```




### Use
  1. prepare csv file of your experimental data
  1. enter working folder name
  1. upload the csv file by either drag&drop or "Browse files" button
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


<br />
<br />


## Uninstall
Below are examples, in case if you want to uninstall or re-install. 

To uninstall (remove all items prepared in above installation), one needs to (1) uninstall MySQL, (2) remove conda environment, (3) delete local repository.  

### Uninstall MySQL

<details><summary>Ubuntu</summary><div>  

1. stop(if running) and remove MySQL server  
	```bash
	sudo systemctl stop mysql
	sudo apt remove --purge mysql-server
	sudo apt remove --purge mysql-client
	sudo apt remove --purge mysql-common
	```
1. remove left database files(if they still exist)
	```bash 
	sudo rm -r /etc/mysql /var/lib/mysql
	```
1. remove other dependencies packages
	```bash 
	sudo apt autoremove --purge
	sudo rm -rf ~/.my.cnf
	```
</div></details>


<details><summary>CentOS7</summary><div>  

1. list of mysql-related items installed  
	```bash
	pm -qa | grep -i mysql  
	```

1. stop mysql server(if running) and remove those
	```bash
	sudo systemctl stop mysqld
	sudo yum remove mysql*
	sudo rm -rf /var/lib/mysql
	sudo rm -rf ~/.my.cnf
	```
</div>
</details>


<details><summary>MacOS(Intel)</summary><div> 

Here, we assume that it was installed via Homebrew
1. stop mysql server(if running) and uninstall mysql
	```bash
	mysql.server stop
	brew uninstall mysql
	```
	
2. remove all directries and files (ignore some of them when they do not exist)  
	```bash
	sudo rm -rf /usr/local/Cellar/mysql*
	sudo rm -rf /usr/local/bin/mysql*
	sudo rm -rf /usr/local/var/mysql*
	sudo rm -rf /usr/local/etc/my.cnf
	sudo rm -rf /usr/local/share/mysql*
	sudo rm -rf /usr/local/opt/mysql*
	sudo rm -rf ~/.my.cnf
	```
3. make sure if anything is left
	```bash
	brew doctor
	```

</div>
</details>

<br />

### Remove conda environment
```bash
conda remove -n mat_interp --all
```
default name of the environment is 'mat_interp'

<br />

### Delete local repository
delete mat_interp directory from your pc