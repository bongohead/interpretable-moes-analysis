# Install R & Rstudio server
mkdir -p /etc/apt/keyrings
wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
  | tee /etc/apt/keyrings/cran.asc

echo "deb [signed-by=/etc/apt/keyrings/cran.asc] \
https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" \
 | tee /etc/apt/sources.list.d/cran.list

apt update -qq
apt install -y --no-install-recommends r-base r-base-dev

# Set locales
apt-get install -y locales
sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
locale-gen
update-locale LANG=en_US.UTF-8

# Install kernel
R -e "install.packages('IRkernel', Ncpus = 8); IRkernel::installspec(user = FALSE)"

jupyter kernelspec list

apt-get install -y libxml2-dev libfontconfig1-dev libcurl4-openssl-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libssl-dev 
R -e "install.packages('tidyverse', Ncpus = 8);"
R -e "install.packages('data.table', Ncpus = 8);"
R -e "install.packages('slider', Ncpus = 8);"

# Fonts for plotting
apt install fonts-cmu 
R -e "install.packages('extrafont', Ncpus = 8);"
R -e "install.packages('svglite', Ncpus = 8);"
R -e "install.packages('ggtext', Ncpus = 8);"

# Fix issue with list-cols not displaying
R -e "install.packages('remotes'); remotes::install_github('IRkernel/repr'); IRkernel::installspec()"

# Rstudio setup - need to expose port 8887 in runpod TCP
# apt-get install -y gdebi-core
# wget https://download2.rstudio.org/server/jammy/amd64/rstudio-server-2025.05.0-496-amd64.deb
# apt-get update -qq
# apt-get install -y ./rstudio-server-2025.05.0-496-amd64.deb
# rm ./rstudio-server-2025.05.0-496-amd64.deb

# useradd -m rstudio
# passwd rstudio

# echo "www-port=8887" >> /etc/rstudio/rserver.conf
# systemctl enable --now rstudio-server

# # Make sure it's up
# ss -ltnp | grep 8887