# getting docker base image
FROM dokken/ubuntu-20.04
# # Curl command doesn't work because docker containers generally don't have access to the internet so I am copying vscode from my OG machine. 
# COPY vscode.deb /tmp

# # To install any other dependencies
# RUN apt-get install -y xdg-utils 

# RUN dpkg -i /tmp/vscode.deb 

# # To install any other dependencies
# RUN apt-get -f install -y

RUN apt-get install -y software-properties-common

RUN add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update

RUN apt-get install -y python3.10

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2

RUN update-alternatives --config python3

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

RUN python3 get-pip.py 

# install Git
RUN apt-get update && apt-get install -y git

RUN rm -rf ~/NLP_RL_Docker_Version

RUN git clone -b docker-branch https://github.com/tirthankarCU/PPO_Docker_Gridworld.git

# changing mount point
WORKDIR /PPO_Docker_Gridworld

# installing python packages.
RUN pip install -r metadata/requirements.txt

# installing additional python packages.
RUN pip install -r metadata/requirements1.txt
